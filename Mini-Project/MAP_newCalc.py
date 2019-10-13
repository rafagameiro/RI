import numpy as np
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer

import collectionloaders
import RetrievalModelsMatrix

# Computes the MAP throughout a specific range of miu values
class MAP_newCalc:

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
        news = np.arange(50, 1500, 50)

        # For each value in the news array it will compute the MAP
        # After going through all the values in the news array
        # it will present a plot with the variation of the MAP throughout the new values
        for new in news:
            models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer, 0.5, new)

            i = 1
            map_model = 0
            for query in cranfield.queries:
                # Parse the query and compute the document scores
                scores = models.score_lmd(parser.stemSentence(query))

                # Do the evaluation
                [average_precision, precision, self.recall, thresholds] = cranfield.eval(scores, i)

                map_model = map_model + average_precision
                i = i + 1

            self.map_val.append(map_model / cranfield.num_queries)
            index = index + 1

        plt.plot(news, self.map_val, color='green', alpha=1)
        plt.ylim([0.27, 0.30])
        plt.xlim([0.0, 1500.0])
        plt.xlabel('New')
        plt.ylabel('MAP')
        plt.title('MAP-New')
        plt.savefig('results/map-new.png', dpi=100)
        plt.show()

