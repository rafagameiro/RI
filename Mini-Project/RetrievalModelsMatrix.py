import numpy as np


class RetrievalModelsMatrix:

    def __init__(self, tf, vectorizer):
        self.vectorizer = vectorizer
        self.tf = tf
        self.termCollFreq = np.sum(tf, axis=0)
        self.docLen = np.sum(tf, axis=1)

        self.idf = np.log(np.size(tf, axis = 0) / self.termCollFreq)
        self.tfidf = np.array(tf * self.idf)

        self.docNorms = np.sqrt(np.sum(np.power(self.tfidf, 2), axis=1))
        
    def score_vsm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        query_norm = np.sqrt(np.sum(np.power(query_vector, 2), axis=1))

        doc_scores = np.dot(query_vector, self.tfidf.T) / (self.docNorms * query_norm)

        return doc_scores

    def score_lmd(self, query):
        return 0

    def score_lmjm(self, query):
        return 0

    def score_bm25(self, query):
        return 0

    def scoreRM3(self, query):
        return 0

