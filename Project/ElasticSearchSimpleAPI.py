from elasticsearch import Elasticsearch
from elasticsearch.exceptions import TransportError
import json

from pandas.io.json import json_normalize
import pandas as pd


class ESSimpleAPI:

    def __init__(self):
        self.client = Elasticsearch(hosts=["zarco.fast"])
        self.index="msmarco"
        
    def search_json_results(self, query=None, numDocs=10):
        result = self.client.search(index=self.index, body={"query": {"match": {"body": query}}}, size=numDocs)
        return result
    
    def search_body(self, query=None, numDocs=10):
        result = self.client.search(index=self.index, body={"query": {"match": {"body": query}}}, size=numDocs)
        df = json_normalize(result["hits"]["hits"])
        return df

    def search_entities(self, query=None, numDocs=10):
        result = self.client.search(index=self.index, body={"query": {"match": {"entities": query}}}, size=numDocs)
        df = json_normalize(result["hits"]["hits"])
        return df

    def search_QSL(self, query_qsl=None, numDocs=10):
        result = self.client.search(index=self.index, body=query_qsl, size=numDocs)
        df = json_normalize(result["hits"]["hits"])
        return df

