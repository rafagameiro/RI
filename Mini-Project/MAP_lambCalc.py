import numpy as np
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer

import collectionloaders
import RetrievalModelsMatrix

# Computes the MAP throughout a specific range of lambda values
class MAP_lambCalc:

    def __init__(self, bigrams):

        ### 1. Load the corpus
        cranfield = collectionloaders.CranfieldTestBed()

        ### 2. Parse the corpus
        # Tokenize, stem and remove stop words
        if not bigrams:
            vectorizer = CountVectorizer()
        else:
            vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b',
                                         min_df=1)

        corpus = parser.stemCorpus(cranfield.corpus_cranfield['abstract'])

        ### 3. Create the model
        # Compute the term frequencies matrix and the model statistics
        tf_cranfield = vectorizer.fit_transform(corpus).toarray()
        
        index = 0
        self.map_val = []
        lambs = np.arange(0, 1, 0.1)

        # For each value in the lambs array it will compute the MAP
        # After going through all the values in the lambs array
        # it will present a plot with the variation of the MAP throughout the lamb values
        for lamb in lambs:
            models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer, lamb)

            i = 1
            map_model = 0
            for query in cranfield.queries:
                # Parse the query and compute the document scores
                scores = models.score_lmjm(parser.stemSentence(query))

                # Do the evaluation
                [average_precision, precision, self.recall, thresholds] = cranfield.eval(scores, i)

                map_model = map_model + average_precision
                i = i + 1

            self.map_val.append(map_model / cranfield.num_queries)
            index = index + 1

        plt.plot(lambs, self.map_val, color='b', alpha=1)
        plt.ylim([0.0, 0.5])
        plt.xlim([0.0, 1.0])
        plt.xlabel('Lambda')
        plt.ylabel('MAP')
        plt.title('MAP-Lambda')
        plt.savefig('results/map-lamb.png', dpi=100)
        plt.show()

