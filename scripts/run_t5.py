import argparse
import glob
import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from numbert.utils.transformer_base import BaseTransformer, add_generic_args, generic_train, get_linear_schedule_with_warmup
from numbert.utils.args import ModelArguments, DataProcessingArguments, TrainingArguments
from numbert.utils.data_utils import tf_dl 
from numbert.utils.model_utils.utils_t5 import Seq2SeqRankingDataset as Dataset


logger = logging.getLogger(__name__)


class Seq2SeqRankingTrainer(BaseTransformer):

    mode = "language-modeling"

    def __init__(self, args):
        super().__init__(args, num_labels=None, mode=self.mode)
        self.dataset_kwargs: dict = dict(
            data_dir = args.data_dir,
            max_source_length = args.max_source_length,
            max_target_length = args.max_target_length,
        )

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels,
        )

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        logger.info(batch)
        source_ids, source_mask, y = batch[0], batch[1], batch[4]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,)

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = Dataset.trim_seq2seq_batch(batch, pad_token_id)
        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_cnn.py
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=1,
            max_length=2,
            repetition_penalty=0,
            length_penalty=0,
            early_stopping=True, #TODO check
            use_cache=True,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        loss = self._step(batch)

        return {"val_loss": loss, "preds": preds, "target": target}

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def test_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions.txt")
        output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets.txt")
        # write predictions and targets for later rouge evaluation.
        with open(output_test_predictions_file, "w+") as p_writer, open(output_test_targets_file, "w+") as t_writer:
            for output_batch in outputs:
                p_writer.writelines(s + "\n" for s in output_batch["preds"])
                t_writer.writelines(s + "\n" for s in output_batch["target"])
            p_writer.close()
            t_writer.close()

        return self.test_end(outputs)

    def get_dataloader(self, type_path: str, batch_size: int) -> DataLoader:
        dataset = Dataset(self.tokenizer, type_path=type_path, **self.dataset_kwargs)
        data_set_args = {'batch_size': batch_size,
                         'max_seq_len': self.dataset_kwargs["max_source_length"],
                         'train': type_path == "train",
                         'num_workers': max(self.hparams.num_workers, 1),
                         'seed': self.hparams.seed + self.hparams.local_rank + 1,
                         'threaded_dl':self.hparams.num_workers > 0,
                         'task': self.hparams.task_name,
                         'in_batch_negative':self.hparams.in_batch_negative,
                         'max_tseq_len': self.dataset_kwargs["max_target_length"]
                         }
        dataloader = tf_dl.TFRecordDataLoader(dataset, **data_set_args)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)


def main(args):

    # If output_dir not provided, a folder will be generated in pwd
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)
    model = Seq2SeqRankingTrainer(args)
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