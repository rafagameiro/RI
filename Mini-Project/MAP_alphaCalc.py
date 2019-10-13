import numpy as np
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.feature_extraction.text import CountVectorizer
import collectionloaders
import RetrievalModelsMatrix

# Computes the MAP throughout a specific range of alpha values
# The limit chosen to define the threshold in the RM3 models can also vary
class MAP_alphaCalc:

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

        colors = ['green', 'red', 'blue']
        limits = [-3, -5, -10]

        # For each defined limit it will compute the MAP variation
        # given a specific range of alpha values
        for k in limits:
            index = 0
            self.map_val = []
            alphas = np.arange(0, 1, 0.1)
            for alpha in alphas:
                models = RetrievalModelsMatrix.RetrievalModelsMatrix(tf_cranfield, vectorizer, alpha)

                i = 1
                map_model = 0
                for query in cranfield.queries:
                    # Parse the query and compute the document scores
                    scores = models.scoreRM3(parser.stemSentence(query), k, alpha)

                    # Do the evaluation
                    [average_precision, precision, self.recall, thresholds] = cranfield.eval(scores, i)

                    # Compute the words that were considered relevant in this query
                    words = self.show_query_terms(vectorizer, models)
                    print('\nalpha:', alpha, ', limit:', abs(k), '\n', words)

                    map_model = map_model + average_precision
                    i = i + 1

                self.map_val.append(map_model / cranfield.num_queries)
                index = index + 1

            # Creates the plot that will show the MAP variation with the different alpha values
            plt.plot(alphas, self.map_val, color=colors[limits.index(k)], alpha=1,
                     label='limit = ' + str(abs(limits[limits.index(k)])))

        plt.legend(loc='upper left')
        plt.ylim([0.0, 0.5])
        plt.xlim([0.0, 1.0])
        plt.xlabel('Alpha')
        plt.ylabel('MAP')
        plt.title('MAP-Alpha')
        plt.savefig('results/map-alpha.png', dpi=100)
        plt.show()

    # Goes through vector generated in the RM3 model and checks which positions have values different from zero
    # Those positions means the word was relevant for the query so its stored in a new array that will be returned
    def show_query_terms(self, vectorizer, models):
        query_vector = models.get_rm3()[0]
        words = []
        count = 0
        terms_total = vectorizer.get_feature_names()
        for i in query_vector:
            if i > 0:
                words.append(terms_total[count])
            count = count + 1

        return words
