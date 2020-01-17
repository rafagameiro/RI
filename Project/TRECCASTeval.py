from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import TransportError
import json
import numpy as np

from pandas.io.json import json_normalize
import pandas as pd


class ConvSearchEvaluation:

    def __init__(self):

        with open("/scratch/treccastweb/2019/data/training/train_topics_v1.0.json", "rt", encoding="utf-8") as f: 
            self.train_topics = json.load(f) #dados de treino
            
        # fields: topic_turn_id, docid, rel
        self.relevance_judgments = pd.read_csv("/scratch/treccastweb/2019/data/training/train_topics_mod.qrel", sep=' ', names=["topic_turn_id", "dummy", "docid", "rel"])
                
    def eval(self, result, topic_turn_id):
        
        
        aux = self.relevance_judgments.loc[self.relevance_judgments['topic_turn_id'] == (topic_turn_id)]
        idx_rel_docs = aux.loc[aux['rel'] != 0]
        
        query_rel_docs = idx_rel_docs['docid']
        relv_judg_list = idx_rel_docs['rel']

        #print(idx_rel_docs)- o prof diz que isto e query_rel_docs sao importantes
        
        p10 = 0
        recall = 0
        ndcg = 0
        if query_rel_docs.count() != 0:
            
            # P@10
            top10 = result['_id'][:10]
            true_pos= np.intersect1d(top10,query_rel_docs)
            p10 = np.size(true_pos) / 10

            true_pos= np.intersect1d(result['_id'],query_rel_docs)
            recall = np.size(true_pos) / query_rel_docs.count()

            # Precision-recall raw
            num_ret_docs = result.count()[0]

            # Precision-recall raw
            relev_judg_vector = np.zeros((num_ret_docs,1))
            for rel_doc in query_rel_docs:
                aux = (result['_id'] == rel_doc).to_numpy()
                relev_judg_vector = aux*1 + np.ravel(relev_judg_vector)


            keys = query_rel_docs.keys()
            keys[keys <= 10] = 0
            # Average precision            
            relev_judg_vector = np.zeros((result['_id'].count(),1))            
            relev_judg_vector[keys,0] = (relv_judg_list>0) 

            average_precision = average_precision_score(relev_judg_vector, result['_score'].T)

            #print(average_precision)


            # Precision-recall raw


            # Interpolated Precision-recall


            # Normalized Discount Cummulative Gain




            # 11-point interpolated Precision-recall (the same as treceval)

        
        return [p10, recall, ndcg]


