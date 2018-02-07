from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.cluster import k_means
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import random
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from data_cleaning_before_modeling import *
from sklearn.model_selection import train_test_split
from extra_cleaning_obj1 import *
import itertools
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import silhouette_score, silhouette_samples


def prepare_data_for_KMEANS(data_path):
    data = pulling_data(data_path)
    data['text_reviews'][data['text_reviews'].isnull()] = ''
    data['former_current'][data['former_current'].isnull()] = 1

    data['all_text'] = data['review_titles'] + data['text_reviews']
    data['all_text'][data['all_text'].isnull()] = ''
    data = data.drop(columns=['user_ids', 'review_titles', 'text_reviews', 'position', 'city'], axis=1)

    Text = data['all_text']
    vectorizer = CountVectorizer(stop_words='english', tokenizer=tokenize, max_features= 1000)
    vector = vectorizer.fit_transform(Text).todense()

    return vector


def silhouette(X):
    sil_dict = {}
    for no_k in range(2,20):
        model = KMeans(n_clusters=no_k)
        y_pred = model.fit(X)
        Yhat = y_pred.labels_.reshape(-1,1)
        sil = silhouette_score(X,Yhat)
        sil_dict[no_k] = sil
    return sil_dict

if __name__ == '__main__':
    vector = prepare_data_for_KMEANS('data/Google_data.csv')
    sil = silhouette(vector)
    plt.scatter(list(sil.keys()),list(sil.values()))
