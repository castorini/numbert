import os
import pickle
import logging

from torch.utils.data import Dataset

from transformers.tokenization_utils import trim_batch

import tensorflow as tf

from numbert.utils.utils_numbert import (
    processors,
    create_int64_feature,
    compute_metrics
)  

import torch_xla.core.xla_model as xm
import numpy as np
import re
from collections import defaultdict

__all__ = ['Seq2SeqRankingDataset', 'eval_epoch_end']

logger = logging.getLogger(__name__)


class Seq2SeqRankingDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir=".",
        type_path="train",
        max_source_length=512,
        max_target_length=1,
        task_name="msmarco",
        is_duo=False
    ):
        super().__init__()
        processor = processors[task_name]()
        self.task_name = task_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.label_list = processor.get_labels()
        self.is_duo = is_duo
        self.tokenizer = tokenizer
        self.writer_file = os.path.join(data_dir, 'dataset_{}.tf'.format(type_path))
        self.cached_guid_map_file = os.path.join(data_dir, type_path + "_guid_map.p")
        if self.task_name == "treccar":
            self.cached_oq_map_file = os.path.join(data_dir, type_path + "_oq_map.p")

        if is_duo:
            self.pattern = "Query: {query} sentence1: {sentence1} sentence2: {sentence2} </s>"
            self.target_map = {"0": "false </s>", "1": "true </s>"} # TODO sentence1, sentence2 instead
        else:
            self.pattern = "Query: {query} Document: {document} Relevant: </s>"
            self.target_map = {"0": "false </s>", "1": "true </s>"}

        if os.path.exists(os.path.join(data_dir, 'dataset_{}.tf'.format(type_path))):
            with open(self.cached_guid_map_file, 'rb') as fp:
                self.guid_list = pickle.load(fp)
            if task_name == "treccar":
                with open(self.cached_oq_map_file, "rb") as fp:
                    self.original_queries = pickle.load(fp)
        else:
            if type_path != "train": # ignore for train_triples
                logger.info("Loading Collection")
                processor.load_collection(data_dir)
            if type_path == "test":
                self.examples = processor.get_test_examples(data_dir, is_duo = is_duo)
            else:
                self.examples = processor.get_dev_examples(data_dir, is_duo = is_duo) if type_path == "dev" else processor.get_train_examples(data_dir, is_duo = is_duo)
            if self.task_name == "treccar":
                (self.examples, self.original_queries) = self.examples
            self.writer = tf.io.TFRecordWriter(self.writer_file)
            self.encode_features()
            with open(self.cached_guid_map_file, "wb") as fp:
                pickle.dump(self.guid_list, fp)
            if self.task_name == "treccar":
                with open(self.cached_oq_map_file, "wb") as fp:
                    pickle.dump(self.original_queries, fp)

    def __len__(self):
        return len(self.guid_list)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}

    def encode_features(self, batch_size=10000):
        self.guid_list = []
        self.label_map = {label : i for i, label in enumerate(self.label_list)}

        for (ex_index, example) in enumerate(self.examples):
            if ex_index%batch_size == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(self.examples)))
                if self.is_duo:
                    batch_examples = list(map(
                        lambda ex: self.pattern.format(query = ex.text_a, 
                                                       sentence1 = ex.text_b, 
                                                       sentence2 = ex.text_c), 
                        self.examples[ex_index:ex_index + batch_size]))
                else:
                    batch_examples = list(map(
                        lambda ex: self.pattern.format(query = ex.text_a, 
                                                       document = ex.text_b), 
                        self.examples[ex_index:ex_index + batch_size]))
                if ex_index == 0:
                    logger.info(batch_examples[:5])
                    target_token_map = {label: self.tokenizer.encode(text) \
                                        for label, text in self.target_map.items()}
                batch_source_tokens = self.tokenizer.batch_encode_plus(
                    batch_examples, 
                    pad_to_max_length = True, 
                    return_attention_mask = True, 
                    max_length = self.max_source_length,
                    return_token_type_ids = False
                )
                batch_target_ids = list(map(
                    lambda ex: target_token_map[ex.label], 
                    self.examples[ex_index:ex_index + batch_size]))

            input_ids = batch_source_tokens["input_ids"][ex_index%batch_size]
            if input_ids[-1] != self.tokenizer.pad_token_id: #Make sure last source token is eos
                input_ids[-1] = self.tokenizer.eos_token_id
            attention_mask = batch_source_tokens["attention_mask"][ex_index%batch_size]
            target_ids = batch_target_ids[ex_index%batch_size]
            label = self.label_map[example.label]

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("source_input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("target_ids: %s" % " ".join([str(x) for x in target_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            self.guid_list.append(example.guid)
            fdict = {
                'input_ids': create_int64_feature(input_ids),
                'attention_mask': create_int64_feature(attention_mask),
                'labels': create_int64_feature([label]),
                'guid': create_int64_feature([ex_index]),
                'target_ids': create_int64_feature(target_ids)
            }
            if self.task_name == "treccar":
                if ex_index <= 5:
                    print("TREC")
                fdict['len_gt_titles'] = create_int64_feature([example.len_gt_titles])

            tf_features = tf.train.Features(feature=fdict)
            tf_example = tf.train.Example(features=tf_features)
            self.writer.write(tf_example.SerializeToString())

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id): #TODO check batch numbers?
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"])
        return source_ids, source_mask, y

def eval_epoch_end(outputs, guid_list, args, original_queries = None, split = "dev"):
    # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
    results = {}
    for res_type in ["preds", "out_label_ids", "lguids", "llen_gt_titles"]:
        if res_type == "llen_gt_titles" and args.task_name != "treccar":
            continue
        else:
            results[res_type] = np.concatenate([x[res_type] for x in outputs], axis=0)
            results[res_type] = xm.mesh_reduce("eval_" + res_type, results[res_type], np.concatenate)

    lguids = list(map(lambda x: tuple(re.split(r'-',guid_list[x[0]])), list(results["lguids"])))
    numbert_predictions = defaultdict(defaultdict)
    numbert_labels = defaultdict(set)
    for ind, guid in enumerate(lguids):
        if results["out_label_ids"][ind] == 1:
            numbert_labels[guid[1]].add(guid[3])
        # each guid[3] has a scores for each paired doc having guid[3] in left
        if guid[3] in numbert_predictions[guid[1]]:
            numbert_predictions[guid[1]][guid[3]] += results["preds"][ind]
        else:
            numbert_predictions[guid[1]][guid[3]] = results["preds"][ind]

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
            if results["llen_gt_titles"][ind] > len(numbert_labels[guid[1]]):
                # Metrics like NDCG and MAP require the total number of relevant docs.
                # The code below adds missing number of relevant docs to gt so the 
                # metrics are the same as if we had used all ground-truths.
                # The extra_gts have all negative ids so they don't interfere with the
                # predicted ids, which are all equal or greater than zero.
                extra_gts = list(-(np.arange(max(0, results["llen_gt_titles"][ind] - len(numbert_labels[guid[1]])) + 1)))
                numbert_labels[guid[1]].update(extra_gts)
    metric_score = compute_metrics(args.task_name, numbert_predictions_no_score, numbert_labels)
    if xm.is_master_ordinal():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        if args.trec_output:
            output_treccar_file = os.path.join(args.output_dir, "treccar_predictions_" + split + ".tsv")

            with open(output_treccar_file, "w") as f_trec:
                for query_id in numbert_predictions:
                    rank = 1
                    for doc_id, score in numbert_predictions[query_id]:
                        f_trec.write(" ".join((original_queries[int(query_id)], "Q0", doc_id, str(rank), str(score), "BERT")) + "\n")
                        rank += 1

        if args.msmarco_output:
            output_msmarco_file = os.path.join(args.output_dir, "msmarco_predictions_" + split + ".tsv")
            with open(output_msmarco_file, "w") as f_msmarco:
                for query_id in numbert_predictions:
                    rank = 1
                    for doc_id, _ in numbert_predictions[query_id]:
                        f_msmarco.write("\t".join((query_id, doc_id, str(rank))) + "\n")
                        rank += 1

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(metric_score.keys()):
                logger.info("  %s = %s", key, str(metric_score[key]))
                writer.write("%s = %s\n" % (key, str(metric_score[key])))
