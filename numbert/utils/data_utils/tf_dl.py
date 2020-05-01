# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch DataLoader for TFRecords"""

import queue
import threading

import tensorflow as tf
import torch

class TFRecordDataLoader(object):
    def __init__(self, records, batch_size, max_seq_len, train, num_workers=2, seed=42, threaded_dl=False, task="msmarco", 
                 in_batch_negative=False, rank = -1, num_shards=8, max_tseq_len=None, length=0):
        tf.random.set_seed(seed)
        if isinstance(records, str):
            records  = [records]

        record_format = {"input_ids": tf.io.FixedLenFeature([max_seq_len], tf.int64),
                         "attention_mask": tf.io.FixedLenFeature([max_seq_len], tf.int64),
                         "labels" : tf.io.FixedLenFeature([1], tf.int64),
                         "guid" : tf.io.FixedLenFeature([1], tf.int64)}

        if max_tseq_len == None: #classification
            record_format["token_type_ids"] = tf.io.FixedLenFeature([max_seq_len], tf.int64)
        else: # seq2seq (like T5)
            record_format["target_ids"] = tf.io.FixedLenFeature([max_tseq_len], tf.int64)

        if task == "treccar":
            record_format["len_gt_titles"] = tf.io.FixedLenFeature([1], tf.int64)

        self.record_converter = Record2Example(record_format)
        self.length = length
        #Instantiate dataset according to original BERT implementation
        if train and (not in_batch_negative):
            self.dataset = tf.data.Dataset.from_tensor_slices(tf.constant(records))
            self.dataset = self.dataset.shuffle(buffer_size=len(records))

            # use sloppy tfrecord dataset
            self.dataset = self.dataset.apply(
                tf.data.experimental.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=train,
                    cycle_length=min(num_workers, len(records))))
            self.dataset = self.dataset.shuffle(buffer_size=100)
        else:
            self.dataset = tf.data.TFRecordDataset(records)

        if rank!=-1:
            self.dataset = self.dataset.shard(num_shards=num_shards, index=rank)

        # Instantiate dataloader (do not drop remainder for eval)
        loader_args = {'batch_size': batch_size, 
                       'num_parallel_batches': num_workers,
                       'drop_remainder': train}
        self.dataloader = self.dataset.apply(tf.data.experimental.map_and_batch(self.record_converter, **loader_args))
        self.threaded_dl = threaded_dl
        self.num_workers = num_workers

    def __iter__(self):
        if self.threaded_dl:
            data_iter = iter(MultiprocessLoader(self.dataloader, self.num_workers))
            for item in data_iter:
                yield item
        else:
            data_iter = iter(self.dataloader)
            for item in data_iter:
                yield convert_tf_example_to_torch_tensors(item)

    def __len__(self):
        return self.length


class Record2Example(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, record):
        """Decodes a BERT TF record to a TF example."""
        example = tf.io.parse_single_example(record, self.feature_map)
        for k, v in list(example.items()):
            if v.dtype == tf.int64:
                example[k] = tf.cast(v, tf.int64) #TODO check why it was int32 originally
        return example

def convert_tf_example_to_torch_tensors(example):
    return {k: torch.from_numpy(v.numpy()) for k,v in example.items()}

class MultiprocessLoader(object):
    def __init__(self, dataloader, num_workers=2):
        self.dl = dataloader
        self.queue_size = 2*num_workers

    def __iter__(self):
        output_queue = queue.Queue(self.queue_size)
        output_thread = threading.Thread(target=_multiproc_iter,
                                         args=(self.dl, output_queue))
        output_thread.daemon = True
        output_thread.start()

        while output_thread.is_alive():
            yield output_queue.get(block=True)
        else:
            print(RuntimeError('TF record data loader thread exited unexpectedly'))

def _multiproc_iter(dl, output_queue):
    data_iter = iter(dl)
    for item in data_iter:
        tensors = convert_tf_example_to_torch_tensors(item)
        output_queue.put(tensors, block=True)
