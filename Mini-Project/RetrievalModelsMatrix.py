import numpy as np


class RetrievalModelsMatrix:

    def __init__(self, tf, vectorizer, lamb=0.5, new=100):
        self.vectorizer = vectorizer
        self.tf = tf

        ## VSM statistics
        self.term_doc_freq = np.sum(tf != 0, axis=0)
        self.term_coll_freq = np.sum(tf, axis=0)
        self.docLen = np.sum(tf, axis=1)

        self.idf = np.log(np.size(tf, axis=0) / self.term_doc_freq)
        self.tfidf = np.array(tf * self.idf)

        self.docNorms = np.sqrt(np.sum(np.power(self.tfidf, 2), axis=1))

        smoothing = 0.01
        ## LMD statistics

        # Creates a matrix with the size of term_doc_freq size where every position
        # will have the total length of the documents
        self.ptmd = np.dot(np.ones((np.size(self.term_doc_freq), 1)), [self.docLen])

        # Computes the Probability of a term given the corpus
        # It can me translated into the total number of words in the document
        # divided by the sum of of words in all documents
        self.ptmc = self.term_coll_freq / sum(self.term_coll_freq)

        # Computes the LMD matrix according to the given expression
        self.lmd = (self.tf + new * self.ptmc) / (self.ptmd + new).T

        ## LMJM statistics

        # Computes the Probability of a term given the document
        # Since some documents might be empty a smoothing is applied
        self.ptmd = self.tf * (1 / (self.ptmd.T + smoothing))

        # Computes the LMJM matrix according to the given expression
        self.lmjm = lamb * self.ptmd + (1 - lamb) * self.ptmc

        ## RM3 statistics
        self.prm3 = []


    def score_vsm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        query_norm = np.sqrt(np.sum(np.power(query_vector, 2), axis=1))

        doc_scores = np.dot(query_vector, self.tfidf.T) / (0.0001 + self.docNorms * query_norm)

        return doc_scores

    # Computes the score of the query using the LMD model
    def score_lmd(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()

        doc_scores = np.prod(self.lmd ** query_vector, axis=1)

        #doc_scores = np.sum(np.log(self.lmd) * query_vector, axis=1)

        return doc_scores

    # Computes the score of the query using the LMJM model
    def score_lmjm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()

        doc_scores = np.prod(self.lmjm ** query_vector, axis=1)

        return doc_scores

    # Computes the score of the query using the RM3 model
    def scoreRM3(self, query, limit, alpha):

        query_vector = self.vectorizer.transform([query]).toarray()

        # Computes the LMJM score using the initial query
        doc_scores = self.score_lmjm(query)

        # Defines a threshold
        # Sorts the computed scores and returns the value in the position limit
        thres = np.sort(doc_scores)[limit]

        # Based on the threshold creates an array based on the computed scores
        # If the position of the score has a value bigger than the threshold value,
        # the value will persist in the new array
        # if it is lesser than the threshold value it will become zero in the new array
        top_doc = np.array(doc_scores * (doc_scores > thres))

        # Computes the Probability of a term in the query given the document
        # Creates a matrix with the size of term_doc_freq size where every row
        # will be a copy of the values the array top_doc
        ptqmd = np.dot(np.ones((np.size(self.term_doc_freq), 1)), [top_doc])

        # Computes the probability of the relevance model 1 according with the given expression
        prm1 = np.sum(ptqmd.T * self.ptmd, axis=0)

        # Computes the Probability of a word given the query
        pwmq = query_vector / np.sum(query_vector, axis=1)

        # Computes the probability of the relevance model 3 according with the given expression
        self.prm3 = (1 - alpha) * pwmq + alpha * prm1

        # Given the new query vector (RM3) recomputes the score for the documents
        doc_scores = np.prod(self.lmjm ** self.prm3, axis=1)

        return doc_scores

    def get_rm3(self, ):
        return self.prm3
