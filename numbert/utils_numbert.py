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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import copy
import json
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from .utils import convert_to_unicode
from transformers.file_utils import is_tf_available
import tensorflow as tf
import collections
from . import metrics
import numpy as np

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label, guid = None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.guid = guid

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class MsmarcoProcessor(DataProcessor):
    """Processor for the MS-MARCO data set."""

    def load_qrels(self, path):
        """Loads qrels into a dict of key: query_id, value: list of relevant doc ids."""
        qrels = collections.defaultdict(set)
        with open(path) as f:
            for i, line in enumerate(f):
                query_id, _, doc_id, relevance = line.rstrip().split('\t')
                if int(relevance) >= 1:
                    qrels[query_id].add(doc_id)
                if i % 1000 == 0:
                    print('Loading qrels {}'.format(i))
        return qrels

    def load_queries(self, path):
        """Loads queries into a dict of key: query_id, value: query text."""
        queries = {}
        with open(path) as f:
            for i, line in enumerate(f):
                query_id, query = line.rstrip().split('\t')
                queries[query_id] = query
                if i % 1000 == 0:
                    print('Loading queries {}'.format(i))
        return queries


    def load_run(self, path):
        """Loads run into a dict of key: query_id, value: list of candidate doc ids."""

        # We want to preserve the order of runs so we can pair the run file with the
        # TFRecord file.
        run = collections.OrderedDict()
        with open(path) as f:
            for i, line in enumerate(f):
                query_id, doc_title, rank = line.split('\t')
                if query_id not in run:
                    run[query_id] = []
                run[query_id].append((doc_title, int(rank)))
                if i % 1000000 == 0:
                    print('Loading run {}'.format(i))
        # Sort candidate docs by rank.
        sorted_run = collections.OrderedDict()
        for query_id, doc_titles_ranks in run.items():
            sorted(doc_titles_ranks, key=lambda x: x[1])
            doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
            sorted_run[query_id] = doc_titles

        return sorted_run

    def load_train_triples(self, path):
        triples = []
        with open(path) as f:
            for i, line in enumerate(f):
                query, positive_doc, negative_doc = line.rstrip().split('\t')
                triples.append((query, positive_doc, negative_doc))
        return triples


    def load_collection(self, data_dir):
        """Loads tsv collection into a dict of key: doc id, value: doc text."""
        self.collection = {}
        print('Loading Collection...')
        with open(os.path.join(data_dir, "collection.tsv")) as f:
            for i, line in enumerate(f):
                doc_id, doc_text = line.rstrip().split('\t')
                self.collection[doc_id] = doc_text.replace('\n', ' ')
                if i % 1000000 == 0:
                    print('Loading collection, doc {}'.format(i))

    def merge(self, qrels, run, queries):
        """Merge qrels and runs into a single dict of key: query, 
        value: tuple(relevant_doc_ids, candidate_doc_ids)"""
        data = collections.OrderedDict()
        for query_id, candidate_doc_ids in run.items():
            query = queries[query_id]
            relevant_doc_ids = set()
            if qrels:
                relevant_doc_ids = qrels[query_id]
            data[query_id] = (query, relevant_doc_ids, candidate_doc_ids)
        return data

    def get_train_examples(self, data_dir, is_qrels = None):
        """See base class."""

        return self._create_examples_train_triples(self.load_train_triples(
                os.path.join(data_dir,"triples.train.small.tsv")),
                "train")

    def get_train_examples_old(self, data_dir, is_qrels = True):
        """See base class. Used when not using triples"""
        qrels = None
        if is_qrels:
            qrels = self.load_qrels(os.path.join(data_dir,"qrels.train.tsv"))

        queries = self.load_queries(os.path.join(data_dir,"queries.train.tsv"))
        run = self.load_run(os.path.join(data_dir,"run.train.tsv"))
        train_data = self.merge(qrels=qrels, run=run, queries=queries)

        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir, is_qrels = True):
        """See base class."""
        qrels = None
        if is_qrels:
            qrels = self.load_qrels(os.path.join(data_dir,"qrels.dev.small.tsv"))

        queries = self.load_queries(os.path.join(data_dir,"queries.dev.small.tsv"))
        run = self.load_run(os.path.join(data_dir,"run.dev.small.tsv"))
        dev_data = self.merge(qrels=qrels, run=run, queries=queries)

        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir, is_qrels = True):
        """See base class."""
        qrels = None
        if is_qrels:
            qrels = self.load_qrels(os.path.join(data_dir,"qrels.eval.small.tsv"))

        queries = self.load_queries(os.path.join(data_dir,"queries.eval.small.tsv"))
        run = self.load_run(os.path.join(data_dir,"run.eval.small.expanded-topk10.tsv"))
        eval_data = self.merge(qrels=qrels, run=run, queries=queries)

        return self._create_examples(eval_data, "eval")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples_train_triples(self, data, set_type):
        """Creates examples for the training triples."""
        examples = []
        for (i, triple) in enumerate(data):
            query, doc_p, doc_n = triple
            text_a = convert_to_unicode(query)
            labels = [1, 0]
            for doc_ind, doc in enumerate([doc_p, doc_n]):
                guid = "%s-%s-%s-%s" % (set_type, i, doc_ind, doc_ind)
                text_b = convert_to_unicode(doc)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=str(labels[doc_ind]))) 
        return examples

    def _create_examples(self, data, set_type):
        """Creates examples for the training(2) and dev sets."""
        examples = []
        for (i, query_id) in enumerate(data):
            query, qrels, doc_titles = data[query_id]
            text_a = convert_to_unicode(query)
            doc_titles = doc_titles[:1000] #TODO generalize
            labels = [
              1 if doc_title in qrels else 0 
              for doc_title in doc_titles
            ]
            for doc_ind, doc_title in enumerate(doc_titles):
                guid = "%s-%s-%s-%s" % (set_type, query_id, doc_ind, doc_title)
                text_b = convert_to_unicode(self.collection[doc_title])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=str(labels[doc_ind])))
        return examples

def _create_int64_feature(value):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
  return feature

# TODO should work right now but eventually use encode_plus? 
def convert_examples_to_features(examples, tokenizer,
                                 max_length=512,
                                 task=None,
                                 label_list=None, 
                                 output_mode=None,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True, #Beyond here unused in new transformers
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1, 
                                 cls_token_at_end=False, 
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 max_len_A = 64 - 2,
                                 max_len_B = 448 - 1,
                                 use_tfrecord = False,
                                 writer = None
                                 ):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    guid_list = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)

        # inputs = tokenizer.encode_plus(
        #     example.text_a,
        #     example.text_b,
        #     add_special_tokens=True,
        #     max_length=max_length,
        # )
        # input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # # tokens are attended to.
        # attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)
        tokens = tokens_a[:max_len_A - int(sep_token_extra)] + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b[:max_len_B - int(sep_token_extra)] + [sep_token]
            if sep_token_extra:
                tokens += [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens) - len(segment_ids))

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        if use_tfrecord:
            guid_list.append(example.guid)
            tf_features = tf.train.Features(feature={
                'input_ids': _create_int64_feature(input_ids),
                'attention_mask': _create_int64_feature(attention_mask),
                'token_type_ids': _create_int64_feature(token_type_ids),
                'labels': _create_int64_feature([label]),
                'guid': _create_int64_feature([ex_index])
            })
            tf_example = tf.train.Example(features=tf_features)
            writer.write(tf_example.SerializeToString())

        else:
            features.append(
                    InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  label=label,
                                  guid=example.guid))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))
    if use_tfrecord:
        return guid_list
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "msmarco":
        METRICS = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']
        all_metrics = np.zeros(len(METRICS))
        for key in preds:
            all_metrics += metrics.metrics(
              gt=labels[key], pred=preds[key], metrics_map=METRICS)
        all_metrics /= len(preds)
        METRICS_MAP = {}
        for ind, METRIC in enumerate(METRICS):
            METRICS_MAP[METRIC] = all_metrics[ind]
        return METRICS_MAP
    else:
        raise KeyError(task_name)

processors = {
    "msmarco": MsmarcoProcessor
}

output_modes = {
    "msmarco": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "msmarco": 2
}


