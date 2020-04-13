# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for Passage Ranking."""


import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import re
import torch
import tensorflow as tf
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    get_linear_schedule_with_warmup,
)

from numbert.utils.args import ModelArguments, DataProcessingArguments, TrainingArguments
from numbert.utils.utils_numbert import (
    compute_metrics,
    convert_examples_to_features,
    output_modes, 
    processors,
)  

from numbert.utils.data_utils import tf_dl 


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)

ORIGINAL_QUERIES = None # maintains list of original queries, which are used especially during TREC-CAR output

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, train_guid = None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.use_tfrecord:
        data_set_args = {'batch_size': args.train_batch_size if args.local_rank == -1 \
                                        else args.per_gpu_train_batch_size, # todo check if words for distributed
                         'max_seq_len': args.max_seq_length,
                         'train': True,
                         'num_workers': max(args.num_workers, 1),
                         'seed': args.seed + args.local_rank + 1,
                         'threaded_dl': args.num_workers > 0,
                         'task': args.task_name,
                         'in_batch_negative': args.in_batch_negative
                         }
        train_dataloader = tf_dl.TFRecordDataLoader(train_dataset,
                                                    **data_set_args) #here train dataset is just path to tf record file
    else:
        if args.in_batch_negative:
            train_sampler = SequentialSampler(train_dataset)
        else:
            train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)     
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # Redundant maybe but useful
    if args.use_tfrecord:
        train_per_epoch = len(train_guid)
    else:
        train_per_epoch = len(train_dataloader)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (train_per_epoch // args.gradient_accumulation_steps) + 1
    else:
        t_total = train_per_epoch // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_per_epoch)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss, print_loss = 0.0, 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            if args.use_tfrecord:
                batch = {k: v.to(args.device) for k,v in batch.items()}
            else:
                batch = tuple(t.to(args.device) for t in batch)

            if args.use_tfrecord:
                batch_guids = batch.pop('guid', None)
                if step < 2:
                    logger.info(batch_guids)
                batch_len_gt_titles = batch.pop('len_gt_titles', None)
                if args.model_type == 'distilbert' or args.model_type not in ["bert", "xlnet", "albert"]:
                    _ = batch.pop('token_type_ids', None)
                inputs = batch  
            else:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                batch_guids = batch[4]
                if args.task_name == "treccar":
                    batch_len_gt_titles = batch[5]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print({"step": global_step})

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.print_loss_steps == 0:
                    logging.info("Loss: %f", (tr_loss - print_loss)/args.print_loss_steps)
                    print_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Passage Ranking Evaluation
    log_softmax = torch.nn.LogSoftmax()
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_guid = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)


        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        if args.use_tfrecord:
            data_set_args = {'batch_size': args.eval_batch_size,
                             'max_seq_len': args.max_seq_length,
                             'train': False,
                             'num_workers': max(args.num_workers, 1),
                             'seed': args.seed, #args.seed + args.rank + 1,
                             'threaded_dl': args.num_workers > 0,
                             'task': args.task_name
                             }
            eval_dataloader = tf_dl.TFRecordDataLoader(eval_dataset,
                                                       **data_set_args) #here eval dataset is just path to tf record file
        else:
            eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        if args.use_tfrecord:
            logger.info("  Num examples = %d", len(eval_guid))
        else:
            logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            if args.use_tfrecord:
                batch = {k: v.to(args.device) for k,v in batch.items()}
            else:
                batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                if args.use_tfrecord:
                    batch_guids = batch.pop('guid', None)
                    batch_len_gt_titles = batch.pop('len_gt_titles', None)
                    if args.model_type == 'distilbert' or args.model_type not in ['bert', 'xlnet', "albert"]:
                        _ = batch.pop('token_type_ids', None)
                    inputs = batch  
                else:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    batch_guids = batch[4]
                    if args.task_name == "treccar":
                        batch_len_gt_titles = batch[5]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            log_logits = log_softmax(logits)
            if preds is None:
                preds = log_logits.detach().cpu().numpy()
                lguids = list(map(lambda x: tuple(re.split(r'-',eval_guid[x])) , batch_guids.detach().cpu()))
                if args.task_name == "treccar":
                    llen_gt_titles = batch_len_gt_titles.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, log_logits.detach().cpu().numpy(), axis=0)
                lguids += list(map(lambda x: tuple(re.split(r'-',eval_guid[x])) , batch_guids.detach().cpu()))
                if args.task_name == "treccar":
                    llen_gt_titles = np.append(llen_gt_titles, batch_len_gt_titles.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        numbert_predictions = {}
        numbert_labels = {}
        for ind, guid in enumerate(lguids):
            if out_label_ids[ind] == 1:
                if guid[1] in numbert_labels:
                    numbert_labels[guid[1]].add(guid[3])
                else:
                    numbert_labels[guid[1]] = {guid[3]}
            if guid[1] in numbert_predictions:
                # each guid[3] has a scores for each paired doc having guid[3] in left
                if guid[3] in numbert_predictions[guid[1]]:
                    numbert_predictions[guid[1]][guid[3]] += preds[ind][1]
                else:
                    numbert_predictions[guid[1]][guid[3]] = preds[ind][1]
            else:
                numbert_predictions[guid[1]] = {}
                numbert_predictions[guid[1]][guid[3]] = preds[ind][1]

        # converts {a:b, c:d} to [(a,b), (c,d)]
        for qid in numbert_predictions:
            numbert_predictions[qid] = list(numbert_predictions[qid].items())  
        numbert_predictions_no_score = {}
        for key in numbert_predictions:
            numbert_predictions[key] = sorted(numbert_predictions[key], key=lambda tup: tup[1], reverse=True)
            numbert_predictions_no_score[key] =  list(map(lambda x: x[0], numbert_predictions[key]))
            if key not in numbert_labels:
                numbert_labels[key] = set()

        if args.task_name == "treccar":
            for ind, guid in enumerate(lguids):
                if llen_gt_titles[ind] > len(numbert_labels[guid[1]]):
                    # Metrics like NDCG and MAP require the total number of relevant docs.
                    # The code below adds missing number of relevant docs to gt so the 
                    # metrics are the same as if we had used all ground-truths.
                    # The extra_gts have all negative ids so they don't interfere with the
                    # predicted ids, which are all equal or greater than zero.
                    extra_gts = list(-(np.arange(max(0, llen_gt_titles[ind] - len(numbert_labels[guid[1]])) + 1)))
                    numbert_labels[guid[1]].update(extra_gts)

        eval_loss = eval_loss / nb_eval_steps
        result = compute_metrics(eval_task, numbert_predictions_no_score, numbert_labels)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")

        if args.trec_output:
            split = "test" if args.do_test else "dev"
            output_treccar_file = os.path.join(eval_output_dir, "treccar_predictions_" + split + ".tsv")

            with open(output_treccar_file, "w") as f_trec:
                for query_id in numbert_predictions:
                    rank = 1
                    for doc_id, score in numbert_predictions[query_id]:
                        f_trec.write(" ".join((ORIGINAL_QUERIES[int(query_id)], "Q0", doc_id, str(rank), str(score), "BERT")) + "\n")
                        rank += 1

        if args.msmarco_output:
            split = "eval" if args.do_test else "dev"
            output_msmarco_file = os.path.join(eval_output_dir, "msmarco_predictions_" + split + ".tsv")
            with open(output_msmarco_file, "w") as f_msmarco:
                for query_id in numbert_predictions:
                    rank = 1
                    for doc_id, _ in numbert_predictions[query_id]:
                        f_msmarco.write("\t".join((query_id, doc_id, str(rank))) + "\n")
                        rank += 1

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    guid_list = []
    writer = None
    dataset = None
    global ORIGINAL_QUERIES

    split = 'dev' if evaluate else 'train'
    if args.do_test:
        split = 'eval'
    dataset = os.path.join(args.data_dir, 'dataset_{}.tf'.format(split))

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        split,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    cached_guid_map_file = cached_features_file + "_guid_map.p"
    if args.task_name == "treccar":
        cached_oq_map_file = cached_features_file + "_oq_map.p"
    if (os.path.exists(cached_features_file) and not args.overwrite_cache) and (not args.use_tfrecord):
        assert os.path.exists(cached_guid_map_file)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        with open(cached_guid_map_file, 'rb') as fp:
            guid_list = pickle.load(fp)
        if args.task_name == "treccar":
            with open(cached_oq_map_file, "rb") as fp:
                ORIGINAL_QUERIES = pickle.load(fp)

    elif args.use_tfrecord and os.path.exists(os.path.join(args.data_dir, 'dataset_{}.tf'.format(split))): #TODO caching
        with open(cached_guid_map_file, 'rb') as fp:
            guid_list = pickle.load(fp)
        if args.task_name == "treccar":
            with open(cached_oq_map_file, "rb") as fp:
                ORIGINAL_QUERIES = pickle.load(fp)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['msmarco','treccar'] and evaluate: # ignore for train_triples
            logger.info("Loading Collection")
            processor.load_collection(args.data_dir)
        if args.do_test:
            examples = processor.get_test_examples(args.data_dir, is_duoBERT = args.is_duoBERT)
        else:
            examples = processor.get_dev_examples(args.data_dir, is_duoBERT = args.is_duoBERT) if evaluate else processor.get_train_examples(args.data_dir, is_duoBERT = args.is_duoBERT)
        if args.task_name == "treccar":
            (examples, ORIGINAL_QUERIES) = examples
        if args.use_tfrecord:
            writer = tf.io.TFRecordWriter(dataset)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                use_tfrecord = args.use_tfrecord,
                                                writer = writer,
                                                task = args.task_name,
                                                is_duoBERT = args.is_duoBERT,
                                                is_encode_batch = args.encode_batch)
        if args.use_tfrecord:
            guid_list = features
        if args.local_rank in [-1, 0]:
            if not args.use_tfrecord:
                for f in features:
                    guid_list.append(f.guid)
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)
            with open(cached_guid_map_file, "wb") as fp:
                pickle.dump(guid_list, fp)
            if args.task_name == "treccar":
                with open(cached_oq_map_file, "wb") as fp:
                    pickle.dump(ORIGINAL_QUERIES, fp)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if args.use_tfrecord:
        return dataset, guid_list
    else:
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        all_guids = torch.tensor([i for i in range(len(guid_list))], dtype=torch.long)
        if task == "treccar":
            all_len_gt_titles = torch.tensor([f.len_gt_titles for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_guids, all_len_gt_titles)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_guids)
        return dataset, guid_list


def main():
    parser = HfArgumentParser((ModelArguments, DataProcessingArguments, TrainingArguments))
    model_args, dataprocessing_args, training_args = parser.parse_args_into_dataclasses()

    # For now, let's merge all the sets of args into one,
    # but soon, we'll keep distinct sets of args, with a cleaner separation of concerns.
    args = argparse.Namespace(**vars(model_args), **vars(dataprocessing_args), **vars(training_args))

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare Ranking task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, cache_dir=args.cache_dir,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_dataset, train_guid = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, train_guid)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, use_fast=True)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, use_fast=True)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
