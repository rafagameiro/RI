import numpy as np
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer

import collectionloaders
import RetrievalModelsMatrix

# Computes the MAP of a specific model with static parameters
class computeMAP:

    def __init__(self, bigrams, model_type, is_sw=0.05, is_ba=0.95):

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
        self.map_model = 0
        self.precision_model = []
        self.ap_below = 0
        self.better_query = []
        self.worse_query = []
        
        plt.figure(1)
        for query in cranfield.queries:
            # Parse the query and compute the document scores
            scores = self.compute_score(models, model_type, query)

            # Do the evaluation
            [average_precision, precision, self.recall, thresholds] = cranfield.eval(scores, i)

            # If the computed average precision of the query is below a static value (0.05) it will be presented
            if is_sw > average_precision:
                #print('qid =', i, ' AP=', average_precision)
                self.ap_below = self.ap_below + average_precision
                worse_query.append(i)
                
            if is_ba >= average_precision:
                better_query.append(i)

            # Sums all the average_precision values obtained in the different queries
            self.map_model = self.map_model + average_precision
            self.precision_model.append(precision)
            plt.plot(self.recall, precision, color='silver', alpha=0.1)

            i = i + 1

        # Computes the mean value of MAP and
        # the percentage of queries that have an average precision below a static value
        self.map_model = self.map_model / cranfield.num_queries
        self.ap_below = (self.ap_below / cranfield.num_queries) * 100
        
        print('model ', model_type, ' done.')
        print('MAP = ', self.map_model)
        
    def get_stats(self):
        return self.recall, np.mean(self.precision_model, axis=0)
    
    def get_queries(self):
        return self.better_query, self.worse_query

    # Draws the Precision-Recall curve with the values obtain from the iteration o queries
    def prec_rec_plot(self):

        mean_precision = np.mean(self.precision_model, axis=0)
        std_precision = np.std(self.precision_model, axis=0)

        plt.figure(2)
        plt.plot(self.recall, mean_precision, color='b', alpha=1)
        plt.gca().set_aspect('equal', adjustable='box')

        # Display a space of possible values the MAP can vary
        plt.fill_between(self.recall,
                         mean_precision - std_precision,
                         mean_precision + std_precision, facecolor='b', alpha=0.1)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall (MAP={0:0.2f})'.format(self.map_model))

        # Presents the MAP value and the percentage of queries that have an average precision below a static value
        print('MAP =', self.map_model)
        print('Percentage of AP inferior to 0.05 = ', self.ap_below)

        # Save the drawn plot to a figure
        plt.savefig('results/prec-recall.png', dpi=100)
        
        
    # Depending on the value of the variable "type"
    # the program will execute a different model
    def compute_score(self, models, type, query):
        if type == 0:
            return models.score_vsm(parser.stemSentence(query))
        elif type == 1:
            return models.score_lmd(parser.stemSentence(query))
        elif type == 2:
            return models.score_lmjm(parser.stemSentence(query))
        else:
            return models.scoreRM3(parser.stemSentence(query), -3, 0.5)
