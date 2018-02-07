from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import random
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from data_cleaning_before_modeling import *
from sklearn.model_selection import train_test_split
from extra_cleaning_obj1 import *
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss
from extra_cleaning_obj1 import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import statsmodels.formula.api as sm
import pickle

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def sentiment_scores(sentence):
    snt = analyser.polarity_scores(sentence)
    return snt


def with_text_score(date):
    data_dict = {'wlb' : 'balance', 'pbt': 'benefit', 'jsa': 'security', 'mng': 'management', 'cul': 'culture'}
    for k, v in data_dict.items():
        data_path = 'temp_data/full_' + k + date + '.csv'
        data = pd.read_csv(data_path)
        # data = main_cleaning_function(data)
        # data['text_reviews'][data['text_reviews'].isnull()] = ''
        # data['former_current'][data['former_current'].isnull()] = 1
        #
        # data['all_text'] = data['review_titles'] + data['text_reviews']
        # data['all_text'][data['all_text'].isnull()] = ''
        # data = data.drop(columns=['Unnamed: 0.1','user_ids', 'review_titles', 'text_reviews', 'position', 'city'], axis=1)

        data['text_neg_score'] = 0
        data['text_neu_score'] = 0
        data['text_pos_score'] = 0

        data['num_neg_score'] = 0
        data['num_neu_score'] = 0
        data['num_pos_score'] = 0



        for idx, row in data.iterrows():
            text_score = sentiment_scores(row['all_text'])
            data.ix[idx, 'text_neg_score'] = text_score['neg']
            data.ix[idx, 'text_neu_score'] = text_score['neu']
            data.ix[idx, 'text_pos_score'] = text_score['pos']
        col = v + '_sc'
        for idx, row in data.iterrows():
            if row[col] == 1:
                data.ix[idx, 'num_neg_score'] = 1
            elif row[col] == 2:
                data.ix[idx, 'num_neg_score'] = 0.5
                data.ix[idx, 'num_neu_score'] = 0.5
            elif row[col] == 3:
                data.ix[idx, 'num_neu_score'] = 1
            elif row[col] == 4:
                data.ix[idx, 'num_neu_score'] = 0.5
                data.ix[idx, 'num_pos_score'] = 0.5
            elif row[col] == 5:
                data.ix[idx, 'num_pos_score'] = 1

        final_score = 'final_' + k + '_sc'

        data[final_score] = -1*((data['text_neg_score'] + data['num_neg_score'])/2)  + \
                              0*((data['text_neu_score'] + data['num_neu_score'])/2)  + \
                              1*((data['text_pos_score'] + data['num_pos_score'])/2)

        file_out = 'sentiment_data/sentiment_' + k + '_' + date + '.csv'
        data.to_csv(file_out)




if __name__ == '__main__':
    date = 'Jan29'
    with_text_score(date)
