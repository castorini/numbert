import warnings
warnings.simplefilter('ignore', UserWarning)

import argparse
import glob
import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
import torch_xla.core.xla_model as xm

from numbert.utils.transformer_base import BaseTransformer, generic_train, get_linear_schedule_with_warmup
from numbert.utils.args import ModelArguments, DataProcessingArguments, TrainingArguments
from numbert.utils.data_utils import tf_dl 
from numbert.utils.model_utils.utils_t5 import Seq2SeqRankingDataset, eval_epoch_end
from numbert.utils.model_utils.decode import greedy_decode

logger = logging.getLogger(__name__)


class Seq2SeqRankingTrainer(BaseTransformer):

    mode = "language-modeling"

    def __init__(self, args):
        super().__init__(args, num_labels=None, mode=self.mode)
        self.dataset_kwargs: dict = dict(
            data_dir = args.data_dir,
            max_source_length = args.max_source_length,
            max_target_length = args.max_target_length
        )
        self.train_dataset_length = -1

    def forward(self, input_ids, attention_mask=None, lm_labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, lm_labels=lm_labels,
        )

    def _step(self, batch): #TODO check if this works
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["target_ids"]
        lm_labels = y.clone() 
        lm_labels[y == self.tokenizer.pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, lm_labels=lm_labels)
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def _eval_step(self, batch, batch_idx):
        source_ids, source_mask, y = Seq2SeqRankingDataset.trim_seq2seq_batch(batch, self.tokenizer.pad_token_id)
        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_cnn.py
        _, batch_scores = greedy_decode(self.model,
            source_ids,
            length=self.dataset_kwargs["max_target_length"],
            attention_mask=source_mask,
            return_last_logits=True)
        batch_scores = batch_scores[:, [6136, 1176]]
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        result_step = {"lguids": batch_guids.detach().cpu().numpy(),
                       "out_label_ids": batch["labels"].detach().cpu().numpy(),
                       "preds": batch_scores[:,1].detach().cpu().numpy()}
        if self.hparams.task_name == "treccar":
            result_step["llen_gt_titles"] = batch["len_gt_titles"].detach().cpu().numpy()
        return result_step


    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx)
        return {}

    def validation_epoch_end(self, outputs):
        eval_epoch_end(outputs, self.guid_list, self.hparams, self.original_queries, "dev")
        return {}

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx)
        return {}

    def test_epoch_end(self, outputs):
        eval_epoch_end(outputs, self.guid_list, self.hparams, self.original_queries, "test")
        return {}

    def get_dataloader(self, type_path: str, batch_size: int) -> DataLoader:
        if not xm.is_master_ordinal():
            xm.rendezvous("load_and_cache_examples")
        dataset = Seq2SeqRankingDataset(self.tokenizer, type_path=type_path, **self.dataset_kwargs)
        if xm.is_master_ordinal():
            xm.rendezvous("load_and_cache_examples")
        self.guid_list = dataset.guid_list
        data_set_args = {'batch_size': batch_size,
                         'max_seq_len': self.dataset_kwargs["max_source_length"],
                         'train': type_path == "train",
                         'num_workers': max(self.hparams.num_workers, 1),
                         'seed': self.hparams.seed + self.hparams.local_rank + 1,
                         'threaded_dl':self.hparams.num_workers > 0,
                         'task': self.hparams.task_name,
                         'in_batch_negative':self.hparams.in_batch_negative,
                         'max_tseq_len': self.dataset_kwargs["max_target_length"],
                         'rank': xm.get_ordinal()
                         'length': len(self.guid_list)
                         }
        if type_path == "train":
            self.train_dataset_length = len(dataset.guid_list)
        self.original_queries = dataset.original_queries if self.hparams.task_name == "treccar" else None
        dataloader = tf_dl.TFRecordDataLoader(dataset.writer_file, **data_set_args)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.per_gpu_train_batch_size)
        logger.info("Training Dataset of size %d"%self.train_dataset_length)
        t_total = (
            (self.train_dataset_length // (self.hparams.per_gpu_train_batch_size * max(1, self.hparams.n_gpu, self.hparams.n_tpu_cores)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        logger.info(t_total)
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", batch_size=self.hparams.per_gpu_eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.per_gpu_eval_batch_size)


def main(args):

    # If output_dir not provided, a folder will be generated in pwd
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)
    if args.no_cuda:
        logger.info("No CUDA")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    model = Seq2SeqRankingTrainer(args)
    args.num_train_epochs = int(args.num_train_epochs) #compatibility with run_numbert.py
    trainer = generic_train(model, args)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        # See https://github.com/huggingface/transformers/issues/3159
        # pl use this format to create a checkpoint:
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master\
        # /pytorch_lightning/callbacks/model_checkpoint.py#L169
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        trainer.test(model)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataProcessingArguments, TrainingArguments))
    model_args, dataprocessing_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(model_args), **vars(dataprocessing_args), **vars(training_args))
    main(args)