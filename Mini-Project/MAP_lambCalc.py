import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import collectionloaders
import RetrievalModelsMatrix


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

        for lamb in np.arange(0, 1, 0.1):
            models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer, lamb)

            i = 1
            self.map_model = 0
            for query in cranfield.queries:
                # Parse the query and compute the document scores
                scores = models.score_lmjm(parser.stemSentence(query))

                # Do the evaluation
                [average_precision, precision, self.recall, thresholds] = cranfield.eval(scores, i)

                self.map_model = self.map_model + average_precision
                i = i + 1

            self.map_model = self.map_model / cranfield.num_queries
            print("\nMAP = ", self.map_model)
            print("\nlamb = ", lamb)
            plt.plot(lamb, self.map_model, color='b', alpha=1)

    plt.xlabel('Lambda')
    plt.ylabel('MAP')
    plt.title('MAP-Lambda')
    plt.savefig('results/map-lamb.png', dpi=100)
    plt.show()

