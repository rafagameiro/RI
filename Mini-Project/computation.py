import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import collectionloaders
import RetrievalModelsMatrix


class computation:

    def __init__(self, bigrams):

        is_sw = 0.05

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
        models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer)

        ### 4. Run the queries over the corpus
        i = 1
        self.map_model = 0
        self.precision_model = []
        self.ap_below = 0

        plt.subplot(1, 2, 1)
        for query in cranfield.queries:
            # Parse the query and compute the document scores
            scores = models.score_lmjm(parser.stemSentence(query))

            # Do the evaluation
            [average_precision, precision, self.recall, thresholds] = cranfield.eval(scores, i)

            # Some messages...
            if is_sw > average_precision:
                print('qid =', i, ' AP=', average_precision)
                print('query: \n', query)
                print()
                self.ap_below = self.ap_below + average_precision
            
            self.map_model = self.map_model + average_precision
            self.precision_model.append(precision)
            plt.plot(self.recall, precision, color='silver', alpha=0.1)

            i = i + 1

        self.map_model = self.map_model / cranfield.num_queries
        self.ap_below = (self.ap_below / cranfield.num_queries) * 100

    def prec_rec_plot(self):

        mean_precision = np.mean(self.precision_model, axis=0)
        std_precision = np.std(self.precision_model, axis=0)

        plt.subplot(1, 2, 2)
        plt.plot(self.recall, mean_precision, color='b', alpha=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.fill_between(self.recall,
                         mean_precision - std_precision,
                         mean_precision + std_precision, facecolor='b', alpha=0.1)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall (MAP={0:0.2f})'.format(self.map_model))

        print('MAP =', self.map_model)
        print('Percentage of AP inferior to 0.05 = ', self.ap_below)

        plt.savefig('results/prec-recall.png', dpi=100)
