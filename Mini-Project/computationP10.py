import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import collectionloaders
import RetrievalModelsMatrix


class computeP10:

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
        models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer, 0.5, 250)

        ### 4. Run the queries over the corpus
        i = 1
        self.p10_model = 0
        self.precision_model = []

        plt.figure(1)
        for query in cranfield.queries:
            # Parse the query and compute the document scores
            scores = models.score_lmd(parser.stemSentence(query))

            # Do the evaluation
            [average_precision, precision, self.recall, p10] = cranfield.eval(scores, i)

            self.p10_model = self.p10_model + p10
            self.precision_model.append(precision)
            plt.plot(self.recall, precision, color='silver', alpha=0.1)

            i = i + 1

        self.p10_model = self.p10_model / cranfield.num_queries
        print('\nP10 =', self.p10_model)
