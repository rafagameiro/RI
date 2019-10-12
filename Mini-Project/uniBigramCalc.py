import numpy as np
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer

import collectionloaders
import RetrievalModelsMatrix


class computeUniBi:

    def __init__(self, model_type):

        models_names = ["vsm", "lmd", "lmjm", "rm3"]
        cranfield = collectionloaders.CranfieldTestBed()

        corpus = parser.stemCorpus(cranfield.corpus_cranfield['abstract'])

        colors = ['b', 'r']
        labels = ['unigram', 'bigram']

        for k in range(0, 2):

            if k == 0:
                vectorizer = CountVectorizer()
            else:
                vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b',
                                             min_df=1)

            tf_cranfield = vectorizer.fit_transform(corpus).toarray()
            models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer)

            i = 1
            self.map_model = 0
            self.precision_model = []

            for query in cranfield.queries:
                scores = self.compute_score(models, model_type, query)

                [average_precision, precision, self.recall, thresholds] = cranfield.eval(scores, i)

                self.map_model = self.map_model + average_precision
                self.precision_model.append(precision)
                i = i + 1

            self.map_model = self.map_model / cranfield.num_queries
            mean_precision = np.mean(self.precision_model, axis=0)

            plt.plot(self.recall, mean_precision, color=colors[k], alpha=1, label=labels[k])
            plt.gca().set_aspect('equal', adjustable='box')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.legend(loc='upper left')
        plt.title('Precision-Recall (' + models_names[model_type].upper() + ')')
        plt.savefig('results/uni-bi-' + models_names[model_type] + '.png', dpi=100)

    def compute_score(self, models, type, query):
        if type == 0:
            return models.score_vsm(parser.stemSentence(query))
        elif type == 1:
            return models.score_lmd(parser.stemSentence(query))
        elif type == 2:
            return models.score_lmjm(parser.stemSentence(query))
        else:
            return models.scoreRM3( parser.stemSentence(query))
