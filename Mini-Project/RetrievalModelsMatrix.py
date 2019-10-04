import numpy as np


class RetrievalModelsMatrix:

    def __init__(self, tf, vectorizer):
        self.vectorizer = vectorizer
        self.tf = tf

        ## VSM statistics
        self.term_doc_freq = np.sum(tf != 0, axis=0)
        self.term_coll_freq = np.sum(tf, axis=0)
        self.docLen = np.sum(tf, axis=1)

        self.idf = np.log(np.size(tf, axis=0) / self.term_doc_freq)
        self.tfidf = np.array(tf * self.idf)

        self.docNorms = np.sqrt(np.sum(np.power(self.tfidf, 2), axis=1))

        ## LMD statistics
        new = 100

        self.ptmd = np.dot(np.ones((np.size(self.term_doc_freq), 1)), [self.docLen])
        self.ptmc = self.term_doc_freq / sum(self.term_doc_freq)

        self.lmd = (self.tf + new * self.ptmc) / (self.ptmd + new).T

        ## LMJM statistics
        lamb = 0.5
        smoothing = 0.01

        self.ptmd = self.tf * (1 / (self.ptmd.T + smoothing))

        self.lmjm = lamb * self.ptmd + (1 - lamb) * self.ptmc

        ## BM25 statistics

    def score_vsm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        query_norm = np.sqrt(np.sum(np.power(query_vector, 2), axis=1))

        doc_scores = np.dot(query_vector, self.tfidf.T) / (0.0001 + self.docNorms * query_norm)

        return doc_scores

    def score_lmd(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()

        doc_scores = np.prod(self.lmd ** query_vector, axis=1)

        return doc_scores

    def score_lmjm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()

        doc_scores = np.prod(self.lmjm ** query_vector, axis=1)

        return doc_scores

    def score_bm25(self, query):
        return 0

    def scoreRM3(self, query):
        return 0
