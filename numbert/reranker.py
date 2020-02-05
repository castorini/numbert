'''
Module for providing python interface to numBERT re-rankers
'''

from pyserini.search import pysearch

from .utils_reranker import load_model, online_eval, Hit
from .utils.utils_numbert import (processors, output_modes,
                                    convert_examples_to_features)


class Reranker(object):
    '''
    Base class for Passage Ranking Models
    (Defaults to the behavior of Anserini's search)
    Parameters
    ----------
    index_dir : str
        Path to Lucene index directory
    '''
    def __init__(self, index_dir):
        self.simple_searcher = pysearch.SimpleSearcher(index_dir)

    def rerank(self, query, num_hits_bm25=100, threads=1):
        '''
        Parameters
        ----------
        query : str
            Query string
        num_hits_bm25 : int
            Number of hits to return from pyserini's SimpleSearcher
        threads : int
            Maximum number of threads 
            
        Returns
        -------
        results : list of io.anserini.search.SimpleSearcher$Result
            List of document hits returned from search
        '''
        return self.simple_searcher.search(q=query, k=num_hits_bm25, 
                                           threads=threads)

    def batch_rerank(self, queries, qids, num_hits_bm25=100, threads=1):
        '''
        Parameters
        ----------
        queries : list of str
            list of query strings
        qids : list of str
            list of corresponding query ids
        num_hits_bm25 : int
            Number of hits to return
        threads : int
            Maximum number of threads 
            
        Returns
        -------
        result_dict : dict of {str : io.anserini.search.SimpleSearcher$Result}
            Dictionary of {qid : document hits} returned from each query
        '''
        return self.simple_searcher.batch_search(
            queries=queries, qids=qids, k=num_hits_bm25, threads=threads)

class MonoBERT(Reranker):
    '''
    Class for the monoBERT Ranking Model
    Parameters
    ----------
    index_dir : str
        Path to Lucene index directory
    model_dir : str
        Path to PyTorch model directory
    model_type : str
        Type of pretrained Transformer model used
    task_name : str
        Task being tested (msmarco / treccar)
    max_seq_len : int
        Maximum length of input sequence fed into the model
    batch_size : int
        Batch size used while evaluating re-ranker
    '''
    def __init__(self, index_dir, model_dir = None, model_type="bert", 
                 task_name = "msmarco", max_seq_len = 512, batch_size = 1):
        super().__init__(index_dir)
        self.mono_model, self.mono_model_tokenizer, self.device = load_model(
            model_dir, task_name=task_name)
        self.processor = processors[task_name]()
        self.output_mode = output_modes[task_name]
        self.mono_model_type = model_type
        self.mono_max_seq_len = max_seq_len
        self.mono_batch_size = batch_size

    def rerank(self, query, num_hits_bm25=100, num_hits_mono=10, threads=1):
        '''
        Parameters
        ----------
        query : str
            Query string
        num_hits_bm25 : int
            Number of hits to return from pyserini's SimpleSearcher
        num_hits_mono : int
            Number of hits to return from MonoBERT re-ranker
        threads : int
            Maximum number of threads 
            
        Returns
        -------
        results : list of Hits
            List of document hits returned from search
        '''
        return self.batch_rerank(queries=[query], qids=['0'], 
                                 num_hits_bm25=num_hits_bm25, 
                                 num_hits_mono=num_hits_mono,
                                 threads=threads)['0']

    def batch_rerank(self, queries, qids, num_hits_bm25=100, 
                     num_hits_mono=10, threads=1):
        '''
        Parameters
        ----------
        queries : list of str
            list of query strings
        qids : list of str
            list of corresponding query ids
        num_hits_bm25 : int
            Number of hits to return from pyserini's SimpleSearcher
        num_hits_mono : int
            Number of hits to return from MonoBERT re-ranker
        threads : int
            Maximum number of threads 
            
        Returns
        -------
        result_dict : dict of {str : Hits}
            Dictionary of {qid : document hits} returned from each query
        '''
        batch_hits = super().batch_rerank(queries=queries, qids=qids, 
                                          num_hits_bm25=num_hits_bm25, 
                                          threads=1)
        query_dict = dict(zip(qids, queries))
        examples, docid_dict = self.processor.get_examples_online(
            query_dict, batch_hits)
        features = convert_examples_to_features(
            examples,
            self.mono_model_tokenizer,
            label_list=self.processor.get_labels(),
            max_length=self.mono_max_seq_len,
            pad_on_left=bool(self.mono_model_type in ['xlnet']),
            pad_token=0,
            pad_token_segment_id=4 if self.mono_model_type in ['xlnet'] else 0,
            cls_token_segment_id=2 if self.mono_model_type in ['xlnet'] else 1,
            cls_token_at_end=bool(self.mono_model_type in ['xlnet']),
            is_duoBERT=False,
            is_tokenizers=(self.mono_model_type == "bert"),
            is_encode_batch=(self.mono_model_type == "bert"))

        # Evaluate
        model_preds = online_eval(self.mono_model, self.mono_model_tokenizer, 
                                  self.device, features, self.mono_batch_size)
        preds = {}
        for qid in model_preds:
            for pred in model_preds[qid]:
                if qid in preds:
                    preds[qid].append(Hit(pred[0], 
                        docid_dict[pred[0]], pred[1]))
                else:
                    preds[qid] = [Hit(pred[0], 
                        docid_dict[pred[0]], pred[1])]
            preds[qid] = preds[qid][:num_hits_mono]
        return preds

class DuoBERT(MonoBERT):
    '''
    Class for the duoBERT Ranking Model
    Parameters
    ----------
    index_dir : str
        Path to Lucene index directory
    mono_model_dir : str
        Path to PyTorch mono model directory
    duo_model_dir : str
        Path to PyTorch duo model directory
    mono_model_type : str
        Type of pretrained Transformer model used for mono Ranker
    duo_model_type : str
        Type of pretrained Transformer model used for duo Ranker
    mono_max_seq_len : int
        Maximum length of input sequence fed into the mono model
    duo_max_seq_len : int
        Maximum length of input sequence fed into the duo model
    mono_batch_size : int
        Batch size used while evaluating mono re-ranker
    duo_batch_size : int
        Batch size used while evaluating duo re-ranker
    '''
    def __init__(self, index_dir, mono_model_dir, duo_model_dir= None,
                 mono_model_type="bert", duo_model_type="bert",
                 task_name = "msmarco",
                 mono_max_seq_len = 512, duo_max_seq_len=512, 
                 mono_batch_size=1, duo_batch_size=1):
        super().__init__(index_dir, mono_model_dir, mono_model_type, task_name,
            mono_max_seq_len, mono_batch_size)
        self.duo_model, self.duo_model_tokenizer, self.duo_device = load_model(
            duo_model_dir, task_name=task_name)
        self.duo_model_type = duo_model_type
        self.duo_max_seq_len = duo_max_seq_len
        self.duo_batch_size = duo_batch_size

    def rerank(self, query, num_hits_bm25=100, num_hits_mono=10,
               num_hits_duo=5, threads=1):
        '''
        Parameters
        ----------
        query : str
            Query string
        num_hits_bm25 : int
            Number of hits to return from pyserini's SimpleSearcher
        num_hits_mono : int
            Number of hits to return from MonoBERT re-ranker
        num_hits_duo : int
            Number of hits to return from DuoBERT re-ranker
        threads : int
            Maximum number of threads  
            
        Returns
        -------
        results : list of Hits
            List of document hits returned from search
        '''
        return self.batch_rerank(queries=[query], qids=['0'], 
                                 num_hits_bm25=num_hits_bm25, 
                                 num_hits_mono=num_hits_mono,
                                 num_hits_duo=num_hits_duo,
                                 threads=threads)['0']

    def batch_rerank(self, queries, qids, num_hits_bm25=100, num_hits_mono=10,
                     num_hits_duo=5, threads=1):
        '''
        Parameters
        ----------
        queries : list of str
            list of query strings
        qids : list of str
            list of corresponding query ids
        num_hits_bm25 : int
            Number of hits to return from pyserini's SimpleSearcher
        num_hits_mono : int
            Number of hits to return from MonoBERT re-ranker
        num_hits_duo : int
            Number of hits to return from DuoBERT re-ranker
        threads : int
            Maximum number of threads 
            
        Returns
        -------
        result_dict : dict of {str : Hits}
            Dictionary of {qid : document hits} returned from each query
        '''
        mono_batch_hits = super().batch_rerank(queries=queries, qids=qids,
                                               num_hits_bm25=num_hits_bm25,
                                               num_hits_mono=num_hits_mono, 
                                               threads=threads)
        query_dict = dict(zip(qids, queries))
        examples, docid_dict = self.processor.get_examples_online(
            query_dict, mono_batch_hits, is_duoBERT=True)
        features = convert_examples_to_features(
            examples,
            self.duo_model_tokenizer,
            label_list=self.processor.get_labels(),
            max_length=self.duo_max_seq_len,
            pad_on_left=bool(self.duo_model_type in ['xlnet']),
            pad_token=self.duo_model_tokenizer.convert_tokens_to_ids(
                [self.duo_model_tokenizer.pad_token])[0] \
                if self.duo_model_type != 'bert' else 0,
            pad_token_segment_id=4 if self.duo_model_type in ['xlnet'] else 0,
            cls_token_segment_id=2 if self.duo_model_type in ['xlnet'] else 1,
            cls_token_at_end=bool(self.duo_model_type in ['xlnet']),
            is_duoBERT=True,
            is_tokenizers=(self.duo_model_type == 'bert'),
            is_encode_batch=(self.duo_model_type == 'bert'))

        # Evaluate
        model_preds = online_eval(self.duo_model, self.duo_model_tokenizer, 
                                  self.duo_device, features, 
                                  self.duo_batch_size, is_duoBERT=True)
        preds = {}
        for qid in model_preds:
            for pred in model_preds[qid]:
                if qid in preds:
                    preds[qid].append(Hit(pred[0], 
                        docid_dict[pred[0]], pred[1]))
                else:
                    preds[qid] = [Hit(pred[0], 
                        docid_dict[pred[0]], pred[1])]
            preds[qid] = preds[qid][:num_hits_duo]
        return preds