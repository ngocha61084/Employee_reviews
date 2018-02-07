import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

def take_topic_data_out(date, topic):
    data_path = 'sentiment_data/sentiment_' + topic + '_' + date + '.csv'
    data = pd.read_csv(data_path)
    date = data[['year_2011', 'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018']]
    date = date.idxmax(axis=1)
    date = pd.DataFrame(date)
    var_name = 'year_' + topic
    sc_name = 'final_' + topic + '_sc'
    date.columns = [var_name]
    for idx, row in date.iterrows():
        date.ix[idx,var_name] = int(row[var_name][-4:])
    data = data.drop(columns = ['year_2011', 'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018'],axis=1)
    data = pd.concat([data, date], axis=1)
    data = data[['user_ids', 'company_name', sc_name, var_name]]
    return data


def main_data_preparation(date):
    wlb = take_topic_data_out(date, 'wlb')
    pbt = take_topic_data_out(date, 'pbt')
    jsa = take_topic_data_out(date, 'jsa')
    mng = take_topic_data_out(date, 'mng')
    cul = take_topic_data_out(date, 'cul')
    data = wlb.merge(pbt, how='outer', right_on=['user_ids', 'company_name'], left_on=['user_ids', 'company_name'])
    data = data.merge(jsa, how='outer', right_on=['user_ids', 'company_name'], left_on=['user_ids', 'company_name'])
    data = data.merge(mng, how='outer', right_on=['user_ids', 'company_name'], left_on=['user_ids', 'company_name'])
    data = data.merge(cul, how='outer', right_on=['user_ids', 'company_name'], left_on=['user_ids', 'company_name'])
    year = data[['year_wlb', 'year_pbt', 'year_jsa', 'year_mng', 'year_cul']]
    year = pd.DataFrame(year.mean(axis=1))
    year.columns = ['year']
    data = data.drop(columns = ['year_wlb', 'year_pbt', 'year_jsa', 'year_mng', 'year_cul'], axis=1)
    data = pd.concat([data, year], axis=1)
    data = missing_imputation_part1(data)
    wlb = missing_imputation_part2(data, 'wlb')
    pbt = missing_imputation_part2(data, 'pbt')
    jsa = missing_imputation_part2(data, 'jsa')
    mng = missing_imputation_part2(data, 'mng')
    cul = missing_imputation_part2(data, 'cul')
    data = data.merge(wlb, left_on = ['company_name', "year"], how = 'left', right_on = ['company_name', "year"])
    data = data.merge(pbt, left_on = ['company_name', "year"], how = 'left', right_on = ['company_name', "year"])
    data = data.merge(jsa, left_on = ['company_name', "year"], how = 'left', right_on = ['company_name', "year"])
    data = data.merge(mng, left_on = ['company_name', "year"], how = 'left', right_on = ['company_name', "year"])
    data = data.merge(cul, left_on = ['company_name', "year"], how = 'left', right_on = ['company_name', "year"])
    for col in list(data.columns):
        data[col].fillna(999, inplace=True)
    for topic in ['wlb', 'pbt', 'jsa', 'mng', 'cul']:
        final_topic_name = 'final_' + topic + '_sc'
        same_year = topic + '_mean_same_year'
        diff_year = topic + '_mean_diff_year'
        for idx, row in data.iterrows():
            if row[final_topic_name] == 999:
                if row[same_year] == 999:
                    if row[diff_year] == 999:
                        data.ix[idx,final_topic_name] = row["personal_mean"]
                    if row[diff_year] != 999:
                        data.ix[idx,final_topic_name] = (0.3*row["personal_mean"] + 0.07*row[diff_year])/(0.3 + 0.07)
                else:
                    if row[diff_year] == 999:
                        data.ix[idx,final_topic_name] = (0.3*row["personal_mean"] + 0.63*row[same_year])/(0.3 + 0.63)
                    if row[diff_year] != 999:
                        data.ix[idx,final_topic_name] = 0.3*row["personal_mean"] + 0.63*row[same_year] + 0.07*row[diff_year]
    matrix = data[['company_name', 'final_wlb_sc', 'final_pbt_sc', 'final_jsa_sc', 'final_mng_sc', 'final_cul_sc']]
    mean_matrix = matrix.groupby('company_name').mean()
    return pd.DataFrame(mean_matrix)

def missing_imputation_part1(data):
    personal_mean_dat = data[['final_wlb_sc', 'final_pbt_sc', 'final_jsa_sc', 'final_mng_sc', 'final_cul_sc']]
    personal_mean = pd.DataFrame(personal_mean_dat.mean(axis=1))
    personal_mean.columns = ['personal_mean']
    data = pd.concat([data, personal_mean], axis=1)

    #company mean - same year:
    diff_year_data = pd.DataFrame(columns=['company_name', 'year'])
    for topic in ['wlb', 'pbt', 'jsa', 'mng', 'cul']:
        topic_sc = 'final_' + topic + '_sc'
        company_same_year = data[['company_name', 'year', topic_sc]]
        company_same_year_df = company_same_year.groupby(['company_name', "year"]).mean()
        topic_mean_same_year = topic + "_mean_same_year"
        company_same_year_df[topic_mean_same_year] = company_same_year_df[topic_sc]
        company_same_year_df = company_same_year_df.drop(columns = [topic_sc], axis=1)
        company_same_year_df = company_same_year_df.reset_index()
        data = data.merge(company_same_year_df, left_on = ['company_name', "year"], how = 'left', right_on = ['company_name', "year"])
    return data

def missing_imputation_part2(data, topic):
    topic_sc = 'final_' + topic + '_sc'
    topic_mean_different_year = topic + "_mean_diff_year"
    cols = ['company_name','year', topic_mean_different_year]
    new_data = pd.DataFrame(columns=cols)
    for year in ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']:
        company_different_year = data[['company_name', 'year', topic_sc]]
        company_different_year = company_different_year[company_different_year['year'] != int(year)]
        company_different_year = company_different_year[['company_name', topic_sc]]
        company_different_year_df = company_different_year.groupby(['company_name']).mean()
        company_different_year_df['year'] = int(year)
        topic_mean_different_year = topic + "_mean_diff_year"
        company_different_year_df[topic_mean_different_year] = company_different_year_df[topic_sc]
        company_different_year_df = company_different_year_df.drop(columns = [topic_sc], axis=1)
        company_different_year_df = company_different_year_df.reset_index()
        new_data = pd.concat([new_data, company_different_year_df], axis=0)
    return new_data

def convert_to_nparray(date):
    matrix = main_data_preparation(date)
    return np.array(matrix)

if __name__ == '__main__':
    matrix = main_data_preparation('Jan29')
    out_path = 'website/matrix_factorization'
    np.save(outpath, matrix)

'''
company_lst= ['Adobe',
 'Airbnb',
 'Allstate',
 'Apple',
 'Boeing',
 'Cisco',
 'Dell',
 'Expedia',
 'Google',
 'IBM',
 'Indeed',
 'Intel',
 'JLL',
 'KPMG',
 'Kaiser Permanente',
 'Microsoft',
 'NOKIA',
 'Netflix',
 'Nordstrom',
 'Oracle',
 'Qualcomm',
 'Redfin',
 'Salesforce',
 'T-mobile',
 'Tableau',
 'Tesla',
 'Texas Instrument',
 'Twitter',
 'Uber',
 'University of Washington',
 'Workday',
 'Zillow']
'''
