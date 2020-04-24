import os
import pickle
import logging

from torch.utils.data import Dataset

from transformers.tokenization_utils import trim_batch

import tensorflow as tf

from numbert.utils.utils_numbert import (
    processors,
    create_int64_feature
)  

logger = logging.getLogger(__name__)


class Seq2SeqRankingsDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir=".",
        type_path="train",
        max_source_length=512,
        max_target_length=2,
        task_name="msmarco",
        is_duoBERT=False
    ):
        super().__init__()
        processor = processors[task_name]()
        dataset = os.path.join(data_dir, 'dataset_{}.tf'.format(type_path))
        cached_guid_map_file = type_path + "_guid_map.p"
        self.task_name = task_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.label_list = processor.get_labels()
        self.is_duoBERT = is_duoBERT

        if is_duoBERT:
            self.pattern = "Query: {query} sentence1: {sentence1} sentence2: {sentence2} </s>"
            self.target_map = {"0": "false </s>", "1": "true </s>"} # TODO sentence1, sentence2 instead
        else:
            self.pattern = "Query: {query} Document: {document} Relevant: </s>"
            self.target_map = {"0": "false </s>", "1": "true </s>"}



        if self.task_name == "treccar":
            cached_oq_map_file = type_path + "_oq_map.p"
        if os.path.exists(os.path.join(data_dir, 'dataset_{}.tf'.format(type_path))):
            with open(cached_guid_map_file, 'rb') as fp:
                self.guid_list = pickle.load(fp)
            if task_name == "treccar":
                with open(cached_oq_map_file, "rb") as fp:
                    self.original_queries = pickle.load(fp)
        else:
            if type_path != "train": # ignore for train_triples
                logger.info("Loading Collection")
                processor.load_collection(data_dir)
            if type_path == "test":
                self.examples = processor.get_test_examples(data_dir, is_duoBERT = is_duoBERT)
            else:
                self.examples = processor.get_dev_examples(data_dir, is_duoBERT = is_duoBERT) if type_path == "dev" else processor.get_train_examples(data_dir, is_duoBERT = is_duoBERT)
            if self.task_name == "treccar":
                (self.examples, self.original_queries) = self.examples
            self.writer = tf.io.TFRecordWriter(dataset)
            self.encode_features()
            with open(cached_guid_map_file, "wb") as fp:
                pickle.dump(self.guid_list, fp)
            if self.task_name == "treccar":
                with open(cached_oq_map_file, "wb") as fp:
                    pickle.dump(self.original_queries, fp)
        return dataset

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
                if self.is_duoBERT:
                    batch_examples = list(map(
                        lambda ex: self.pattern.format(query = ex.text_a, 
                                                       sentence1 = ex.text_b, 
                                                       sentence2 = ex.text_c), 
                        self.examples[ex_index:ex_index + batch_size]))
                else:
                    batch_examples = list(map(
                        lambda ex: self.pattern.format(query = ex.text_a, 
                                                       sentence1 = ex.text_b), 
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
        y = trim_batch(batch[4], pad_token_id)
        source_ids, source_mask = trim_batch(batch[0], pad_token_id, attention_mask=batch[1])
        return source_ids, source_mask, y
