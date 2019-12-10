import logging
import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)

from transformers import (BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer)
from tqdm import tqdm
import numpy as np
import re

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

def load_model(model_dir, model_type = "bert", do_lower_case = True, task_name = "msmarco", no_cuda = False):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_dir, num_labels = 2, finetuning_task = task_name)
    tokenizer = tokenizer_class.from_pretrained(model_dir, do_lower_case=do_lower_case)
    model = model_class.from_pretrained(model_dir, from_tf=bool('.ckpt' in model_dir), config = config)
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    model.to(device)
    return model, tokenizer, device

def create_dataset(features):
    guid_list = []
    for f in features:
        guid_list.append(f.guid)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    all_guids = torch.tensor([i for i in range(len(guid_list))], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_guids)
    return dataset, guid_list

def online_eval(model, tokenizer, device, features, batch_size = 1, model_type="bert"):
    log_softmax = torch.nn.LogSoftmax()
    eval_dataset, eval_guid = create_dataset(features)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)
    preds = None
    for batch in tqdm(eval_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            batch_guids = batch[4]
            outputs = model(**inputs)
            _, logits = outputs[:2]
        log_logits = log_softmax(logits)
        if preds is None:
            preds = log_logits.detach().cpu().numpy()
            lguids = list(map(lambda x: tuple(re.split(r'-',eval_guid[x])) , batch_guids.detach().cpu()))
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, log_logits.detach().cpu().numpy(), axis=0)
            lguids += list(map(lambda x: tuple(re.split(r'-',eval_guid[x])) , batch_guids.detach().cpu()))
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    numbert_predictions = {}
    # numbert_labels = {}
    for ind, guid in enumerate(lguids):
        # if out_label_ids[ind] == 1:
        #     if guid[1] in numbert_labels:
        #         numbert_labels[guid[1]].add(guid[3])
        #     else:
        #         numbert_labels[guid[1]] = {guid[3]}
        if guid[1] in numbert_predictions:
            numbert_predictions[guid[1]].append((guid[3], preds[ind][1]))
        else:
            numbert_predictions[guid[1]] = [(guid[3], preds[ind][1])]
    # sort
    for key in numbert_predictions:
        numbert_predictions[key] = sorted(numbert_predictions[key], key=lambda tup: tup[1], reverse=True)
    return numbert_predictions

class Hit:
    def __init__(self, docid, content, score):
        self.docid = docid
        self.content = content
        self.score = score