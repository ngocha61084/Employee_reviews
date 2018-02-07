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

def pulling_data(data_path):
    data = pd.read_csv(data_path)
    data = main_cleaning_function(data)
    return data

def remove_duplicates(df):
    df1 = df.drop(columns=['Unnamed: 0'], axis=1)
    return df1.drop_duplicates(keep='first')

def data_append(lst_companies):
    lst_data_path = []
    for company in lst_companies:
        path = 'data/' + company + '_all_data.csv'
        lst_data_path.append(path)

    data = pd.read_csv('data/Google_all_data.csv')
    for p in lst_data_path:
        data = pd.concat([data, pd.read_csv(p)], axis=0)
    return data

def full_raw_data(lst_companies):
    first_full = pd.read_csv('data/Google_all_data.csv')
    first_full = remove_duplicates(first_full)
    for com in lst_company_topic:
        try:
            path = 'data/' + com + '_all_data.csv'
            com_dat = pd.read_csv(path)
            com_dat =  remove_duplicates(com_dat)
            first_full = pd.concat([first_full, com_dat], axis=0)
        except:
            print("Don't have the {} data".format(com))
    file_name = '/Users/hatran/project/galvanize/capstone/temp_data/Feb26_full.csv'
    first_full.to_csv(file_name)
    return first_full


# def sub_sample(lst_companies):
#     data = data_append(lst_companies)
#     sub_data = data.sample(n=10000)
#     sub_data = pd.concat([sub_data, pd.get_dummies(sub_data['company_name'])], axis=1)
#     sub_data = sub_data.drop(columns=['company_name'], axis=1)
#     return sub_data



# def save_all_data_to_csv(lst_companies):
#     sub_data = sub_sample(lst_companies)
#     file_name = '/Users/hatran/project/galvanize/capstone/temp_data/Jan25_data.csv'
#     sub_data.to_csv(file_name)



def take_out_unique_userid(lst_company_topic):
    # Have the full_raw_data:
    for topic in ['cul', 'jsa', 'mng', 'pbt', 'wlb']:
        first_data= pd.read_csv('data/Google_' + topic + '_data.csv')
        first_data = remove_duplicates(first_data)
        first_data = first_data[['user_ids', 'company_name']]
        for company in lst_company_topic:
            try:
                path = 'data/' + company + '_' + topic + '_data.csv'
                data = pd.read_csv(path)
                data = remove_duplicates(data)
                data = data[['user_ids', 'company_name']]
                first_data = pd.concat([first_data, data], axis=0)
            except:
                print("Don't have data of {0} - topic {1}".format(company, topic))
        file_name = '/Users/hatran/project/galvanize/capstone/temp_data/ids_full_' + topic +'.csv'
        first_data.to_csv(file_name)

def full_topic_raw_data(lst_company_topic):
    #full_data = full_raw_data(lst_company_topic)
    full_data = pd.read_csv('temp_data/Jan29_obj2_full_data.csv')
    full_data = full_data.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1)
    take_out_unique_userid(lst_company_topic)
    for topic in ['cul', 'jsa', 'mng', 'pbt', 'wlb']:
        file_name = '/Users/hatran/project/galvanize/capstone/temp_data/ids_full_' + topic +'.csv'
        top_ids = pd.read_csv(file_name)
        top_data = top_ids.merge(full_data, how='inner', left_on = ['company_name', 'user_ids'], right_on = ['company_name', 'user_ids'])
        #file_name = '/Users/hatran/project/galvanize/capstone/temp_data/full_' + topic +'Feb26.csv'
        file_name = '/Users/hatran/project/galvanize/capstone/temp_data/full_' + topic +'Jan29.csv'
        top_data.to_csv(file_name)

if __name__ == '__main__':


    lst_companies = ['Adobe', 'Airbnb', 'Allstate', 'Apple', 'Boeing', 'Cisco', 'Dell', \
                    'Expedia', 'IBM', 'Indeed', 'Intel', 'JLL', 'Kaiser Permanente', 'KPMG', \
                    'Microsoft', 'Netflix', 'NOKIA', 'Nordstrom', 'Oracle', 'Qualcomm', 'Redfin', \
                    'Salesforce', 'T-mobile', 'Tableau', 'Tesla', 'Texas Instrument', 'Twitter', 'Uber', \
                    'University of Washington', 'Workday', 'Zillow']


    # lst_company_topic = ['Adobe', 'Airbnb', 'Allstate', 'Apple', 'Boeing', 'Cisco', \
    #                     'Dell', 'eBay', 'Expedia', 'IBM', 'Indeed', \
    #                     'Intel', 'JLL', 'Kaiser Permanente', 'KPMG', 'Microsoft', \
    #                     'Netflix', 'NOKIA', 'Nordstrom', 'Oracle', 'Qualcomm', 'Redfin', \
    #                     'Salesforce', 'T-mobile', 'Tableau', 'Tesla', 'Texas Instrument', \
    #                     'Twitter', 'Uber', 'University of Washington', 'Workday', 'Zillow']

    save_all_data_to_csv(lst_companies)
    #full_topic_raw_data(lst_company_topic)
    #save_all_data_to_csv(lst_companies)
