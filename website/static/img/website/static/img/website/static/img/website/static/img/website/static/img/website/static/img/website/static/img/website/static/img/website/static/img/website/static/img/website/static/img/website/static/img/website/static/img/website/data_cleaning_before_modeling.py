import pandas as pd
from US_cities_states_library import US_cities_states

def main_cleaning_function(df):
    df = remove_duplicates(df)
    df = shift_null_review(df)
    df = job_titles_cleaning(df)
    df = locations_cleaning(df)
    df = former_current_numeric(df)
    df = position_cleaning(df)
    df = scores_cleaning(df)
    df = dates_cleaning(df)
    df = state_extra_cleaning(df)
    df = company_name_cleaning(df)
    return df

def remove_duplicates(df):
    df1 = df.drop(columns=['Unnamed: 0'], axis=1)
    return df1.drop_duplicates()

def job_titles_cleaning(df):
    df['position'] = df.job_titles.str.split('(').str[0]
    df['former_current'] = df.job_titles.str.split('(').str[1]
    df['former_current'] = df['former_current'].map(lambda x: x[:-5])
    df = df.drop(columns=['job_titles'], axis=1)
    return df


def locations_cleaning(df):
    df['city'] = df.locations.str.split(',').str[0]
    df['state'] = df.locations.str.split(',').str[1]
    df = name_cleaning(df)
    df = df.drop(columns=['locations'], axis=1)
    return df

def name_cleaning(df):

    us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    }
    for k, v in us_state_abbrev.items():
        for idx, row in df.iterrows():
            if row['city'].lower() == k.lower():
                df.ix[idx,'state'] = v
    return df

def former_current_numeric(df):
    former_current_map = {'Former Employee': 1, 'Current Employee': 0}
    df['former_current'] = df['former_current'].map(former_current_map)
    return df

def position_cleaning(df):

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

def scores_cleaning(df):
    for new_var,old_var in zip(['overall_sc',
       'balance_sc', 'benefit_sc', 'security_sc',
       'management_sc', 'culture_sc'],['overall_scores',
       'balance_scores', 'benefit_scores', 'security_scores',
       'management_scores', 'culture_scores']):
        df[new_var] = df[old_var].str.split(':').str[1]
        df[new_var] = df[new_var].map(lambda x: x[:-2])
        df[new_var] = df[new_var].map(lambda x: int(int(x)/20))
        for idx, row in df.iterrows():
            if row[new_var] == 0:
                row[new_var] = row['overall_sc']
    df = df.drop(columns = ['overall_scores',
       'balance_scores', 'benefit_scores', 'security_scores',
       'management_scores', 'culture_scores'], axis=1)
    for col in ['overall_sc',
       'balance_sc', 'benefit_sc', 'security_sc',
       'management_sc', 'culture_sc']:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
        df = df.drop(columns=[col], axis=1)
    return df

def dates_cleaning(df):
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
    for new_date, idx in zip(['month', 'day', 'year'], [0,1,2]):
        df[new_date] = df['dates'].str.split(' ').str[idx]
    df.month = df['month'].map(month_dict)
    df.day = df.day.map(lambda x: int(x[:-1]))
    df.year = df.year.map(lambda x: int(x))
    df = df.drop(columns=['dates'], axis=1)
    df = pd.concat([df, pd.get_dummies(df['month'], prefix='month')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['year'], prefix='year')], axis=1)
    df = df.drop(columns=['day', 'month', 'year'], axis=1)
    return df

def state_extra_cleaning(df):
    df['state'] = df.state.str.lower().str.rstrip().str.lstrip()
    city_to_state_dict, us_state_abbrev_lower, us_state_abbrev = US_cities_states()
    for idx, row in df.iterrows():
        for k, v in us_state_abbrev_lower.items():
            if row['state'] == k:
                df.ix[idx,'state'] = v
            if v in str(row['state']):
                df.ix[idx,'state'] = v
        for k, v in city_to_state_dict.items():
            if row['state'] == k.lower():
                df.ix[idx,'state'] = v
        if row['state'] not in [v for k, v in us_state_abbrev_lower.items()]:
            df.ix[idx,'state'] = 'unknown'
    df['state'] = df.state.str.lower().str.rstrip().str.lstrip()
    df['state'][df['state'].isnull()] = 'unknown'
    for idx, row in df.iterrows():
        if row['state'] not in ['ca', 'wa', 'ny']:
            df.ix[idx, 'state'] = 'other_states'
    df = pd.concat([df, pd.get_dummies(df['state'])], axis=1)

    df = df.drop(columns=['state'], axis=1)
    return df

def company_name_cleaning(df):
#     df = pd.concat([df, pd.get_dummies(df['company_name'])], axis=1)
#     df = df.drop(columns=['company_name'], axis=1)
     return df

def check_null(df):
    return list(df['user_ids'][pd.isnull(df['text_reviews'])])

def insert_null_text(df, uid):
    for idx, row in df.iterrows():
        if row['user_ids'] == uid:
            upper = df[:idx]
            lower = df[idx:]
            lower.text_reviews = lower.text_reviews.shift(-1)
            lower = lower[:-1]
    df = pd.concat([upper,lower], axis=0)
    return df

def shift_null_review(df):
    lst_null = check_null(df)
    formal_lst = lst_null
    i = 0
    while i < len(formal_lst):
        df = insert_null_text(df, lst_null[0])
        lst_null = check_null(df)
        i += 1
    return df
