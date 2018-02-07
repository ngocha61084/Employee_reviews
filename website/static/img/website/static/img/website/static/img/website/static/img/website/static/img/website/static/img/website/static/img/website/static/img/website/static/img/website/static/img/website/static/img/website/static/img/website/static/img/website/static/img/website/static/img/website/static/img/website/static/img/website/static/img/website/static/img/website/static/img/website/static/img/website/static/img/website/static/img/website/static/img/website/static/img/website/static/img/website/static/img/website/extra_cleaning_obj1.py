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
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from extra_cleaning_obj1 import *



def tokenize(doc):
    snowball = SnowballStemmer("english")
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]


def pulling_data(data_path):
    data = pd.read_csv(data_path)
    data = main_cleaning_function(data)
    return data



def preparing_data_logistic_regression(data, num_clusters=6):
    data['text_reviews'][data['text_reviews'].isnull()] = ''
    data['former_current'][data['former_current'].isnull()] = 1

    data['all_text'] = data['review_titles'] + data['text_reviews']
    data['all_text'][data['all_text'].isnull()] = ''
    data = data.drop(columns=['user_ids', 'review_titles', 'text_reviews', 'position', 'city'], axis=1)

    Text = data['all_text']
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize, max_features= 1000)
    vector = vectorizer.fit_transform(Text).todense()

    kmeans = KMeans(n_clusters=num_clusters).fit(vector)
    text_label = kmeans.labels_.reshape([-1,1])
    text_label = pd.DataFrame(text_label)
    text_label.columns = ['text_groups']

    data_no_text = data.drop(columns=['all_text'], axis=1)

    data_with_text = pd.concat([data, text_label], axis=1)
    data_with_text = data_with_text[data_with_text['text_groups'].isnull() == False]
    data_with_text  = pd.concat([data, pd.get_dummies(text_label['text_groups'], prefix = 'group')], axis=1)
    data_with_text  = data_with_text.drop(columns=['all_text'], axis=1)

    data_with_text = data_with_text[data_with_text['group_5'].isnull()==False]
    data_no_text = data_no_text[data_no_text['overall_sc_4'] != 1][data_no_text['overall_sc_3'] != 1][data_no_text['contract'] == 0]
    data_with_text = data_with_text[data_with_text['overall_sc_4'] != 1][data_with_text['overall_sc_3'] != 1][data_with_text['contract'] == 0]

    return data_no_text, data_with_text

def main_functions_clean_after_KMeans(data_path, num_cltrs=8):
    data = pulling_data(data_path)
    data_no_text, data_with_text = preparing_data_logistic_regression(data, num_cltrs)
    return data_no_text, data_with_text


# def train_test_data_splitting_after_KMeans_logistic(data_path, n_groups=6):
#     data_no_text, data_with_text = main_functions_clean_after_KMeans(data_path, n_groups)
#
#     y_no_text = data_no_text['former_current']
#     X_no_text = data_no_text.drop(columns=['former_current', 'Google', 'overall_sc', 'management_sc', 'other_states'], axis=1)
#
#     y_with_text = data_with_text['former_current']
#     X_with_text = data_with_text.drop(columns=['former_current', 'Google', 'overall_sc', 'management_sc', 'other_states', 'group_0'], axis=1)
#
#
#     X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test = train_test_split(X_no_text, y_no_text, test_size=0.4)
#     X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = train_test_split(X_with_text, y_with_text, test_size=0.4)
#
#     return X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
#            X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test
#
# def train_test_data_splitting_after_KMeans(data_path, n_groups=6):
#     data_no_text, data_with_text = main_functions_clean_after_KMeans(data_path, n_groups)
#
#     y_no_text = data_no_text['former_current']
#     X_no_text = data_no_text.drop(columns=['former_current'], axis=1)
#
#     y_with_text = data_with_text['former_current']
#     X_with_text = data_with_text.drop(columns=['former_current'], axis=1)
#
#
#     X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test = train_test_split(X_no_text, y_no_text, test_size=0.2)
#     X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = train_test_split(X_with_text, y_with_text, test_size=0.2)
#
#     return X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
#            X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test


def train_test_data_splitting_after_KMeans_logistic_sub(data_path, n_groups=6):
    data_no_text, data_with_text = main_functions_clean_after_KMeans(data_path, n_groups)

    y_no_text = data_no_text['former_current']
    X_no_text = data_no_text.drop(columns=['former_current', 'benefit_sc_1', 'security_sc_1', \
                                           'overall_sc_1', 'management_sc_1', 'culture_sc_1', 'balance_sc_1', 'other_states',\
                                            'Google', 'engineer', 'month_1', 'year_2016' ], axis=1)

    y_with_text = data_with_text['former_current']
    X_with_text = data_with_text.drop(columns=['former_current', 'benefit_sc_1', 'security_sc_1', \
                                           'overall_sc_1', 'management_sc_1', 'culture_sc_1', 'balance_sc_1', 'other_states',\
                                            'Google', 'engineer', 'month_1', 'year_2016' ], axis=1)


    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test = train_test_split(X_no_text, y_no_text, test_size=0.4)
    X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = train_test_split(X_with_text, y_with_text, test_size=0.4)

    return X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test


def train_test_data_splitting_after_KMeans_sub(data_path, n_groups=6):
    data_no_text, data_with_text = main_functions_clean_after_KMeans(data_path, n_groups)
    #data_no_text = data_no_text[data_no_text['overall_sc'].where((data_no_text['overall_sc']!=3 or data_no_text['overall_sc']!=4) and data_no_text['engineer'] == 1)]
    #data_with_text = data_with_text[data_no_text['overall_sc'].where((data_no_text['overall_sc']!=3 or data_no_text['overall_sc']!=4) and data_no_text['engineer'] == 1)]

    y_no_text = data_no_text['former_current']
    X_no_text = data_no_text.drop(columns=['former_current'], axis=1)

    y_with_text = data_with_text['former_current']
    X_with_text = data_with_text.drop(columns=['former_current'], axis=1)

    X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test = train_test_split(X_no_text, y_no_text, test_size=0.2)
    X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test = train_test_split(X_with_text, y_with_text, test_size=0.2)

    return X_no_text_train, X_no_text_test, y_no_text_train, y_no_text_test, \
           X_with_text_train, X_with_text_test, y_with_text_train, y_with_text_test
