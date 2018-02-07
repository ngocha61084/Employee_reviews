import pandas as pd
from US_cities_states_library import US_cities_states
import pickle as pk
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
import pandas as pd
from US_cities_states_library import US_cities_states


class Model(object):

    def __init__(self):
        self.df=None
        self.X=None
        with open("gbcmodel.pkl",'rb') as m:
            self.gb_model= pk.load(m)

        with open("Kmeans.pkl", "rb") as m2:
            self.kmeans_model= pk.load(m2)

    def instant_comment_to_df(self,comment):
        # comment =     {'Company': [], 'Job_Position': [], 'Status': [], \
        #               'Day': '[]', 'Month': [], 'Year': [], 'City': [], \
        #               'State': '[]', 'Review_Title': [], 'Review': [], \
        #                'Overall_Score': [1,2,3,4,5], 'Work_Life_Balance_Score': [1,2,3,4,5], \
        #                'Benefit_Compensation_Score': [1,2,3,4,5], 'Job_Security_Advancement_Score' : \
        #                [1,2,3,4,5], 'Management_Score': [1,2,3,4,5], 'Culture_Score': \
        #                [1,2,3,4,5]}
        print(comment)
        cols = ['Company',
                 'Job_Position',
                 'Month',
                 'Year',
                 'State',
                 'Review_Title',
                 'Review',
                 'Overall_Score',
                 'Work_Life_Balance_Score',
                 'Benefit_Compensation_Score',
                 'Job_Security_Advancement_Score',
                 'Management_Score',
                 'Culture_Score']
        df=pd.DataFrame([comment], columns=cols)
        return df


    def main_clean_data(self,comment):
        df=self.instant_comment_to_df(comment)
        df=self.user_ids(df)
        df=self.review_titles(df)
        df=self.job_titles(df)
        df=self.locations(df)
        df=self.dates(df)
        df=self.text_reviews(df)
        df=self.overall_scores(df)
        df=self.balance_scores(df)
        df=self.benefit_scores(df)
        df=self.security_scores(df)
        df=self.management_scores(df)
        df=self.culture_scores(df)
        df=self.company_name(df)
        df=self.remove_duplicates(df)
        df=self.shift_null_review(df)
        df=self.job_titles_cleaning(df)
        df=self.locations_cleaning(df)
        df=self.position_cleaning(df)
        df=self.scores_cleaning(df)
        df=self.dates_cleaning(df)
        df=self.state_extra_cleaning(df)
        df=self.company_name_cleaning(df)
        data_no_text, data_with_text = self.preparing_data_logistic_regression(df)

        df_x=data_with_text.drop(columns=['former_current'], axis=1)
        X=df_x.as_matrix()
        return X

    def predict(self,comment):
        X=self.main_clean_data(comment)
        prediction=self.gb_model.predict_proba(X)
        return prediction

    def company_name_cleaning(self,df):
        df['company_name'] = df['Company']
        df = df.drop(columns=['Company'], axis = 1)
        company_lst =   ['Adobe', 'Airbnb', 'Allstate', \
       'Apple', 'Boeing', 'Cisco', 'Dell', 'Expedia', 'Google', 'IBM', \
       'Indeed', 'Intel', 'JLL', 'KPMG', 'Kaiser Permanente', 'Microsoft', \
       'NOKIA', 'Netflix', 'Nordstrom', 'Oracle', 'Qualcomm', 'Redfin', \
       'Salesforce', 'T-mobile', 'Tableau', 'Tesla', 'Texas Instrument', \
       'Twitter', 'Uber', 'University of Washington', 'Workday', 'Zillow']

        for com in company_lst:
            try:
                df[com] = 0
            except:
                print('')
        return df

    def position_cleaning(self,df):

        lead = ['lead', 'manager', 'head', 'senior']
        engineer = ['engineer', 'developer', 'tech', 'scientist', 'code']
        research = ['r&d', 'research', 'lab', 'science']
        data = ['data', 'analyst']
        contract = ['contract', 'intern', 'free', 'student', 'volunteer', 'fellow', 'trainee']
        support = ['driver', 'cook', 'food', 'cater', 'valet']
        assist = ['assist', 'agent', 'scanner', 'tech', 'associate', 'clerk']
        speacialist = ['specialist', 'operator', 'strategist', 'coordinator', 'designer', \
                      'staff', 'hr', 'recruiter', 'it ', ' it', 'trainer', 'administrator', 'counselor']
        test = ['qa ', ' qa', 'test', 'quality', 'rater']
        front = ['sale', 'consultant', 'partner', 'customer']

        big_lst = [lead, engineer, research, data, contract, support, assist, \
                  speacialist]
        df['lead'], df['engineer'], df['research'], df['data'], df['contract'], \
            df['support'], df['support'], df['assist'], df['speacialist'], df['test'], df['front'] \
            = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for idx, row in df.iterrows():
            for l in lead:
                if l in row['position'].lower():
                    df.ix[idx, 'lead'] = 1
            for e in engineer:
                if e in row['position'].lower():
                    df.ix[idx, 'engineer'] = 1
            for r in research:
                if r in row['position'].lower():
                    df.ix[idx, 'research'] = 1
            for d in data:
                if d in row['position'].lower():
                    df.ix[idx, 'data'] = 1
            for c in contract:
                if c in row['position'].lower():
                    df.ix[idx, 'contract'] = 1
            for s in support:
                if s in row['position'].lower():
                    df.ix[idx, 'support'] = 1
            for a in assist:
                if a in row['position'].lower():
                    df.ix[idx, 'assist'] = 1
            for sp in speacialist:
                if sp in row['position'].lower():
                    df.ix[idx, 'speacialist'] = 1
            for t in test:
                if t in row['position'].lower():
                    df.ix[idx, 'test'] = 1
            for f in front:
                if f in row['position'].lower():
                    df.ix[idx, 'front'] = 1
        df = df.drop(columns=['position'], axis = 1)

        return df

    def month_cleaning(self,df):
        month_dict = {
            'January' : 1,
            'February' : 2,
            'March' : 3,
            'April' : 4,
            'May' : 5,
            'June' : 6,
            'July' : 7,
            'August' : 8,
            'September' : 9,
            'October' : 10,
            'November' : 11,
            'December' : 12
        }
        df['month'] = df['Month'].map(month_dict)
        df['month'] = pd.get_dummies(df.month, prefix = 'month')
        df = df.drop(columns = ['Month', 'month'], axis=1)
        month_lst = ['month_' + str(i) for i in range(1,13)]
        try:
            for mon in month_lst:
                df[mon] = 0
        except:
            pass
        return df

    def year_cleaning(self,df):
        df['year'] = df['Year']
        df['year'] = pd.get_dummies(df.month, prefix = 'year')
        year_lst = ['year_' + str(i) for i in range(2011, 2019)]
        df = df.drop(columns=['year', 'Year'], axis=1)
        try:
            for year in year_lst:
                df[year] = 0
        except:
            pass
        return df

    def state_cleaning(self, df):
        df['state'] = df['State']
        df = df.drop(columns=['State'], axis=1)
        
        return df

    def review_titles_cleaning(self, df):

    #
    #
    # def user_ids(self, df):
    #     df['user_ids'] = 'cmp-review-f123fbaf123456f8'
    #     return df
    #
    # def review_titles(self, df):
    #     df['review_titles'] = df['Review_Title']
    #     df = df.drop(columns = ['Review_Title'], axis=1)
    #     return df
    #
    # def job_titles(self, df):
    #     df['job_titles'] = df['Job_Position'] + '(' + df['Status'] + ')'
    #     df = df.drop(columns = ['Job_Position', 'Status'], axis = 1)
    #     return df
    #
    # def locations(self, df):
    #     df['locations'] = df['City'] + ', ' + df['State']
    #     df = df.drop(columns = ['City', 'State'], axis=1)
    #     return df
    #
    # def dates(self, df):
    #     df['dates']= df['Month'] + ' ' + df['Day'] + ', ' + df['Year']
    #     df = df.drop(columns = ['Month', 'Day', 'Year'], axis=1)
    #     return df
    #
    # def text_reviews(self, df):
    #     df['text_reviews'] = df['Review']
    #     df = df.drop(columns = ['Review'], axis=1)
    #     return df
    #
    # def overall_scores(self, df):
    #     # df['Overall_Score'] = int(df['Overall_Score'])*20
    #     # df['overall_scores'] = 'width: ' + str(df['Overall_Score']) + '%;'
    #     # df = df.drop(columns=['Overall_Score'], axis = 1)
    #     df['overall_sc'] = df['Overall_Score']
    #     df = df.drop(columns = ['Overall_Score'], axis=1)
    #     return df
    #
    # def balance_scores(self, df):
    #     # df['Work_Life_Balance_Score'] = int(df['Work_Life_Balance_Score'])*20
    #     # df['balance_scores'] = 'width: ' + str(df['Work_Life_Balance_Score']) + '%;'
    #     # df = df.drop(columns=['Work_Life_Balance_Score'], axis = 1)
    #     df['balance_sc'] = df['Work_Life_Balance_Score']
    #     df = df.drop(columns = ['Work_Life_Balance_Score'], axis=1)
    #     return df
    #
    # def benefit_scores(self, df):
    #     # df['Benefit_Compensation_Score'] = int(df['Benefit_Compensation_Score'])*20
    #     # df['benefit_scores'] = 'width: ' + str(df['Benefit_Compensation_Score']) + '%;'
    #     # df = df.drop(columns=['Benefit_Compensation_Score'], axis = 1)
    #     df['benefit_sc'] = df['Benefit_Compensation_Score']
    #     df = df.drop(columns = ['Benefit_Compensation_Score'], axis=1)
    #     return df
    #
    # def security_scores(self, df):
    #     # df['Job_Security_Advancement_Score'] = int(df['Job_Security_Advancement_Score'])*20
    #     # df['security_scores'] = 'width: ' + str(df['Job_Security_Advancement_Score']) + '%;'
    #     # df = df.drop(columns=['Job_Security_Advancement_Score'], axis = 1)
    #     df['security_sc'] = df['Job_Security_Advancement_Score']
    #     df = df.drop(columns = ['Job_Security_Advancement_Score'], axis=1)
    #     return df
    #
    # def management_scores(self, df):
    #     # df['Management_Score'] = int(df['Management_Score'])*20
    #     # df['management_scores'] = 'width: ' + str(df['Management_Score']) + '%;'
    #     # df = df.drop(columns=['Management_Score'], axis = 1)
    #     df['management_sc'] = df['Management_Score']
    #     df = df.drop(columns = ['Management_Score'], axis=1)
    #     return df
    #
    # def culture_scores(self, df):
    #     # df['Culture_Score'] = int(df['Culture_Score'])*20
    #     # df['culture_scores'] = 'width: ' + str(df['Culture_Score']) + '%;'
    #     # df = df.drop(columns=['Culture_Score'], axis = 1)
    #     df['culture_sc'] = df['Culture_Score']
    #     df = df.drop(columns = ['Culture_Score'], axis=1)
    #     return df
    #
    # def company_name(self, df):
    #     df['company_name'] = df['Company']
    #     df = df.drop(columns=['Company'], axis = 1)
    #     company_lst =   ['Adobe', 'Airbnb', 'Allstate', \
    #    'Apple', 'Boeing', 'Cisco', 'Dell', 'Expedia', 'Google', 'IBM', \
    #    'Indeed', 'Intel', 'JLL', 'KPMG', 'Kaiser Permanente', 'Microsoft', \
    #    'NOKIA', 'Netflix', 'Nordstrom', 'Oracle', 'Qualcomm', 'Redfin', \
    #    'Salesforce', 'T-mobile', 'Tableau', 'Tesla', 'Texas Instrument', \
    #    'Twitter', 'Uber', 'University of Washington', 'Workday', 'Zillow']
    #
    #     for com in company_lst:
    #         try:
    #             df[com] = 0
    #         except:
    #             print('')
    #     return df
    #
    #
    #
    # # def main_cleaning_function_raw(self,df):
    # #     df = remove_duplicates(df)
    # #     df = shift_null_review(df)
    # #     df = job_titles_cleaning(df)
    # #     df = locations_cleaning(df)
    # #     df = former_current_numeric(df)
    # #     df = position_cleaning(df)
    # #     df = scores_cleaning(df)
    # #     df = dates_cleaning(df)
    # #     df = state_extra_cleaning(df)
    # #     df = company_name_cleaning(df)
    # #     return df
    #
    # def remove_duplicates(self,df):
    #     return df.drop_duplicates()
    #
    # def job_titles_cleaning(self,df):
    #     df['position'] = df.job_titles.str.split('(').str[0]
    #     df['former_current'] = df.job_titles.str.split('(').str[1]
    #     df['former_current'] = df['former_current'].map(lambda x: x[:-5])
    #     df = df.drop(columns=['job_titles'], axis=1)
    #     return df
    #
    #
    # def locations_cleaning(self,df):
    #     df['city'] = df.locations.str.split(',').str[0]
    #     df['state'] = df.locations.str.split(',').str[1]
    #     df = name_cleaning(df)
    #     df = df.drop(columns=['locations'], axis=1)
    #     return df
    #
    # def name_cleaning(self,df):
    #
    #     us_state_abbrev = {
    #     'Alabama': 'AL',
    #     'Alaska': 'AK',
    #     'Arizona': 'AZ',
    #     'Arkansas': 'AR',
    #     'California': 'CA',
    #     'Colorado': 'CO',
    #     'Connecticut': 'CT',
    #     'Delaware': 'DE',
    #     'Florida': 'FL',
    #     'Georgia': 'GA',
    #     'Hawaii': 'HI',
    #     'Idaho': 'ID',
    #     'Illinois': 'IL',
    #     'Indiana': 'IN',
    #     'Iowa': 'IA',
    #     'Kansas': 'KS',
    #     'Kentucky': 'KY',
    #     'Louisiana': 'LA',
    #     'Maine': 'ME',
    #     'Maryland': 'MD',
    #     'Massachusetts': 'MA',
    #     'Michigan': 'MI',
    #     'Minnesota': 'MN',
    #     'Mississippi': 'MS',
    #     'Missouri': 'MO',
    #     'Montana': 'MT',
    #     'Nebraska': 'NE',
    #     'Nevada': 'NV',
    #     'New Hampshire': 'NH',
    #     'New Jersey': 'NJ',
    #     'New Mexico': 'NM',
    #     'New York': 'NY',
    #     'North Carolina': 'NC',
    #     'North Dakota': 'ND',
    #     'Ohio': 'OH',
    #     'Oklahoma': 'OK',
    #     'Oregon': 'OR',
    #     'Pennsylvania': 'PA',
    #     'Rhode Island': 'RI',
    #     'South Carolina': 'SC',
    #     'South Dakota': 'SD',
    #     'Tennessee': 'TN',
    #     'Texas': 'TX',
    #     'Utah': 'UT',
    #     'Vermont': 'VT',
    #     'Virginia': 'VA',
    #     'Washington': 'WA',
    #     'West Virginia': 'WV',
    #     'Wisconsin': 'WI',
    #     'Wyoming': 'WY',
    #     }
    #     for k, v in us_state_abbrev.items():
    #         for idx, row in df.iterrows():
    #             if row['city'].lower() == k.lower():
    #                 df.ix[idx,'state'] = v
    #     return df
    #
    # def former_current_numeric(self, df):
    #     former_current_map = {'Former Employee': 1, 'Current Employee': 0}
    #     df['former_current'] = df['former_current'].map(former_current_map)
    #     return df
    #
    def position_cleaning(self,df):

        lead = ['lead', 'manager', 'head', 'senior']
        engineer = ['engineer', 'developer', 'tech', 'scientist', 'code']
        research = ['r&d', 'research', 'lab', 'science']
        data = ['data', 'analyst']
        contract = ['contract', 'intern', 'free', 'student', 'volunteer', 'fellow', 'trainee']
        support = ['driver', 'cook', 'food', 'cater', 'valet']
        assist = ['assist', 'agent', 'scanner', 'tech', 'associate', 'clerk']
        speacialist = ['specialist', 'operator', 'strategist', 'coordinator', 'designer', \
                      'staff', 'hr', 'recruiter', 'it ', ' it', 'trainer', 'administrator', 'counselor']
        test = ['qa ', ' qa', 'test', 'quality', 'rater']
        front = ['sale', 'consultant', 'partner', 'customer']

        big_lst = [lead, engineer, research, data, contract, support, assist, \
                  speacialist]
        df['lead'], df['engineer'], df['research'], df['data'], df['contract'], \
            df['support'], df['support'], df['assist'], df['speacialist'], df['test'], df['front'] \
            = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for idx, row in df.iterrows():
            for l in lead:
                if l in row['position'].lower():
                    df.ix[idx, 'lead'] = 1
            for e in engineer:
                if e in row['position'].lower():
                    df.ix[idx, 'engineer'] = 1
            for r in research:
                if r in row['position'].lower():
                    df.ix[idx, 'research'] = 1
            for d in data:
                if d in row['position'].lower():
                    df.ix[idx, 'data'] = 1
            for c in contract:
                if c in row['position'].lower():
                    df.ix[idx, 'contract'] = 1
            for s in support:
                if s in row['position'].lower():
                    df.ix[idx, 'support'] = 1
            for a in assist:
                if a in row['position'].lower():
                    df.ix[idx, 'assist'] = 1
            for sp in speacialist:
                if sp in row['position'].lower():
                    df.ix[idx, 'speacialist'] = 1
            for t in test:
                if t in row['position'].lower():
                    df.ix[idx, 'test'] = 1
            for f in front:
                if f in row['position'].lower():
                    df.ix[idx, 'front'] = 1
        #df = df.drop(columns=['position'], axis = 1)
        return df

    # def scores_cleaning(self,df):
    #     # for new_var,old_var in zip(['overall_sc',
    #     #    'balance_sc', 'benefit_sc', 'security_sc',
    #     #    'management_sc', 'culture_sc'],['overall_scores',
    #     #    'balance_scores', 'benefit_scores', 'security_scores',
    #     #    'management_scores', 'culture_scores']):
    #     #     df[new_var] = df[old_var].str.split(':').str[1]
    #     #     df[new_var] = df[new_var].map(lambda x: x[:-2])
    #     #     df[new_var] = df[new_var].map(lambda x: int(int(x)/20))
    #     #     for idx, row in df.iterrows():
    #     #         if row[new_var] == 0:
    #     #             row[new_var] = row['overall_sc']
    #     # df = df.drop(columns = ['overall_scores',
    #     #    'balance_scores', 'benefit_scores', 'security_scores',
    #     #    'management_scores', 'culture_scores'], axis=1)
    #     for col in ['overall_sc',
    #        'balance_sc', 'benefit_sc', 'security_sc',
    #        'management_sc', 'culture_sc']:
    #         df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    #         df = df.drop(columns=[col], axis=1)
    #     return df
    #
    # def dates_cleaning(self,df):
    #     month_dict = {
    #         'January' : 1,
    #         'February' : 2,
    #         'March' : 3,
    #         'April' : 4,
    #         'May' : 5,
    #         'June' : 6,
    #         'July' : 7,
    #         'August' : 8,
    #         'September' : 9,
    #         'October' : 10,
    #         'November' : 11,
    #         'December' : 12
    #     }
    #     for new_date, idx in zip(['month', 'day', 'year'], [0,1,2]):
    #         df[new_date] = df['dates'].str.split(' ').str[idx]
    #     df.month = df['month'].map(month_dict)
    #     df.day = df.day.map(lambda x: int(x[:-1]))
    #     df.year = df.year.map(lambda x: int(x))
    #     df = df.drop(columns=['dates'], axis=1)
    #     df = pd.concat([df, pd.get_dummies(df['month'], prefix='month')], axis=1)
    #     df = pd.concat([df, pd.get_dummies(df['year'], prefix='year')], axis=1)
    #     df = df.drop(columns=['day', 'month', 'year'], axis=1)
    #     return df
    #
    # def state_extra_cleaning(self,df):
    #     df['state'] = df.state.str.lower().str.rstrip().str.lstrip()
    #     city_to_state_dict, us_state_abbrev_lower, us_state_abbrev = US_cities_states()
    #     for idx, row in df.iterrows():
    #         for k, v in us_state_abbrev_lower.items():
    #             if row['state'] == k:
    #                 df.ix[idx,'state'] = v
    #             if v in str(row['state']):
    #                 df.ix[idx,'state'] = v
    #         for k, v in city_to_state_dict.items():
    #             if row['state'] == k.lower():
    #                 df.ix[idx,'state'] = v
    #         if row['state'] not in [v for k, v in us_state_abbrev_lower.items()]:
    #             df.ix[idx,'state'] = 'unknown'
    #     df['state'] = df.state.str.lower().str.rstrip().str.lstrip()
    #     df['state'][df['state'].isnull()] = 'unknown'
    #     for idx, row in df.iterrows():
    #         if row['state'] not in ['ca', 'wa', 'ny']:
    #             df.ix[idx, 'state'] = 'other_states'
    #     df = pd.concat([df, pd.get_dummies(df['state'])], axis=1)
    #
    #     df = df.drop(columns=['state'], axis=1)
    #     return df
    #
    # def company_name_cleaning(self,df):
    #       df = pd.concat([df, pd.get_dummies(df['company_name'])], axis=1)
    #       df = df.drop(columns=['company_name'], axis=1)
    #       return df
    #
    # def check_null(self,df):
    #     return list(df['user_ids'][pd.isnull(df['text_reviews'])])
    #
    # def insert_null_text(self, df, uid):
    #     for idx, row in df.iterrows():
    #         if row['user_ids'] == uid:
    #             upper = df[:idx]
    #             lower = df[idx:]
    #             lower.text_reviews = lower.text_reviews.shift(-1)
    #             lower = lower[:-1]
    #     df = pd.concat([upper,lower], axis=0)
    #     return df
    #
    # def shift_null_review(self,df):
    #     lst_null = check_null(df)
    #     formal_lst = lst_null
    #     i = 0
    #     while i < len(formal_lst):
    #         df = insert_null_text(df, lst_null[0])
    #         lst_null = check_null(df)
    #         i += 1
    #     return df
    #
    #
    #
    #
    # def preparing_data_logistic_regression(self, df):
    #     df['text_reviews'][df['text_reviews'].isnull()] = ''
    #     df['former_current'][df['former_current'].isnull()] = 1
    #
    #     df['all_text'] = df['review_titles'] + df['text_reviews']
    #     df['all_text'][df['all_text'].isnull()] = ''
    #     df = df.drop(columns=['user_ids', 'review_titles', 'text_reviews', 'position', 'city'], axis=1)
    #
    #     Text = df['all_text']
    #     vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize, max_features= 1000)
    #     vector = vectorizer.fit_transform(Text).todense()
    #
    #     #kmeans = KMeans(n_clusters=6).fit(vector)
    #     text_label = self.kmeans_model.labels_.reshape([-1,1])
    #     text_label = pd.DataFrame(text_label)
    #     text_label.columns = ['text_groups']
    #
    #     data_no_text = df.drop(columns=['all_text'], axis=1)
    #
    #     data_with_text = pd.concat([df, text_label], axis=1)
    #     data_with_text = data_with_text[data_with_text['text_groups'].isnull() == False]
    #     data_with_text  = pd.concat([df, pd.get_dummies(text_label['text_groups'], prefix = 'group')], axis=1)
    #     data_with_text  = data_with_text.drop(columns=['all_text'], axis=1)
    #
    #     data_with_text = data_with_text[data_with_text['group_5'].isnull()==False]
    #     # data_no_text = data_no_text[data_no_text['overall_sc_4'] != 1][data_no_text['overall_sc_3'] != 1][data_no_text['contract'] == 0]
    #     # data_with_text = data_with_text[data_with_text['overall_sc_4'] != 1][data_with_text['overall_sc_3'] != 1][data_with_text['contract'] == 0]
    #
    #     return data_no_text, data_with_text
    #
    #
    #
    #
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
