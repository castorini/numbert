'''
Module for providing python interface to numBERT searchers
'''

import logging
from pyserini.search import pysearch
logger = logging.getLogger(__name__)

from .utils_search import load_model, online_eval, Hit
from ..utils_numbert import processors, output_modes, convert_examples_to_features

class MonoBERTSearcher:
    '''
    Parameters
    ----------
    '''
    def __init__(self, index_dir, model_dir = None, model_type="bert", max_seq_len = 512, batch_size = 1):
        self.simple_searcher = pysearch.SimpleSearcher(index_dir)
        self.model, self.model_tokenizer, self.device = load_model(model_dir)
        self.processor = processors["msmarco"]
        self.output_mode = output_modes["msmarco"]
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def search(self, q, k=10, t=-1):
        '''
        Parameters
        ----------
        q : str
            Query string
        k : int
            Number of hits to return
        t : int
            Query tweet time for searching tweets  
            
        Returns
        -------
        results : list of io.anserini.search.SimpleSearcher$Result
            List of document hits returned from search
        '''
        return self.batch_search(queries=[q], qids=['0'], 
                                    k=k, t=t, threads=1)['0']

    def batch_search(self, queries, qids, k=10, t=-1, threads=1):
        '''
        Parameters
        ----------
        queries : list of str
            list of query strings
        qids : list of str
            list of corresponding query ids
        k : int
            Number of hits to return
        t : int
            Query tweet time for searching tweets  
        threads : int
            Maximum number of threads 
            
        Returns
        -------
        result_dict : dict of {str : io.anserini.search.SimpleSearcher$Result}
            Dictionary of {qid : document hits} returned from each query
        '''
        batch_hits = self.simple_searcher.batch_search(queries=queries, qids=qids, k=k,
                                             t=t, threads=1)
        query_dict = dict(zip(qids, queries))
        examples, docid_dict = self.processor.get_examples_online(query_dict, batch_hits)
        features = convert_examples_to_features(examples,
                                                self.tokenizer,
                                                label_list=self.processor.get_labels(),
                                                max_length=self.max_seq_len,
                                                pad_on_left=bool(self.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0,
                                                cls_token_segment_id=2 if self.model_type in ['xlnet'] else 1,
                                                cls_token_at_end=bool(self.model_type in ['xlnet']))
        
        # Evaluate
        model_preds = online_eval(self.model, self.model_tokenizer, self.device, features, self.batch_size)
        preds = {}
        for qid in model_preds:
            for pred in model_preds[qid]:
                if qid in preds:
                    preds[qid].append([Hit(pred[0], docid_dict[pred[0]], pred[1])])
                else:
                    preds[qid] = [Hit(pred[0], docid_dict[pred[0]], pred[1])]
        return preds