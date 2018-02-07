import time
from selenium.webdriver import Firefox
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from collections import OrderedDict
from random import randint
import boto3

s3 = boto3.resource('s3')

def main_scraping_function(company):
    all_link_lst, work_life_balance_lst, pay_benefit_lst, job_secured_lst, management_lst, culture_lst = get_links(company)
    try:
        all_data = create_topic_data(company, all_link_lst)
        export_data_to_csv(convert_to_dataframe(all_data), company, 'all')

    except:
        print('error at {0} with {1}'.format(company, 'all'))

    try:
        wlb_data = create_topic_data(company, work_life_balance_lst)
        export_data_to_csv(convert_to_dataframe(wlb_data), company, 'wlb')
    except:
        print('error at {0} with {1}'.format(company, 'wlb'))

    try:
        pbt_data = create_topic_data(company, pay_benefit_lst)
        export_data_to_csv(convert_to_dataframe(pbt_data), company, 'pbt')
    except:
        print('error at {0} with {1}'.format(company, 'pbt'))

    try:
        jsa_data = create_topic_data(company, job_secured_lst)
        export_data_to_csv(convert_to_dataframe(jsa_data), company, 'jsa')
    except:
        print('error at {0} with {1}'.format(company, 'jsa'))

    try:
        mng_data = create_topic_data(company, management_lst)
        export_data_to_csv(convert_to_dataframe(mng_data), company, 'mng')
    except:
        print('error at {0} with {1}'.format(company, 'mng'))

    try:
        cul_data = create_topic_data(company, culture_lst)
        export_data_to_csv(convert_to_dataframe(cul_data), company, 'cul')
    except:
        print('error at {0} with {1}'.format(company, 'cul'))



def create_topic_data(company, links_lst):
    company_dict     = {}
    content_dict     = {}
    userid_lst       = []
    review_title_lst = []
    job_title_lst    = []
    location_lst     = []
    dates_lst        = []
    text_review_lst  = []
    overalls_lst     = []
    balances_lst     = []
    benefits_lst     = []
    security_lst     = []
    management_lst   = []
    culture_lst      = []
    for link in links_lst:
        browser = Firefox()
        browser.get(link)
        userid_lst += get_user_id(browser)
        review_title_lst += get_title(browser)
        job_title_lst += get_job_title(browser)
        location_lst += get_job_location(browser)
        dates_lst += get_dates(browser)
        text_review_lst += get_text_reviews(browser)
        overalls, balances, benefits, security, management, culture = get_reviews_score(browser)
        overalls_lst += overalls
        balances_lst += balances
        benefits_lst += benefits
        security_lst += security
        management_lst += management
        culture_lst += culture
        ran_nums = randint(5,20)
        time.sleep(np.random.choice(ran_nums,1)[0])
        browser.quit()
        time.sleep(np.random.choice(ran_nums,1)[0])

    content_dict['user_ids'] = userid_lst
    content_dict['review_titles'] = review_title_lst
    content_dict['job_titles'] = job_title_lst
    content_dict['locations'] = location_lst
    content_dict['dates'] = dates_lst
    content_dict['text_reviews'] = text_review_lst
    content_dict['overall_scores'] = overalls_lst
    content_dict['balance_scores'] = balances_lst
    content_dict['benefit_scores'] =  benefits_lst
    content_dict['security_scores'] =  security_lst
    content_dict['management_scores'] =  management_lst
    content_dict['culture_scores'] =  culture_lst
    company_dict[company] = content_dict
    return company_dict



def get_links(company):

    '''
    Purpose: the function wil automatically generate the list of links
    of all tech companies's reviews
    Input: company name
    Output: list of links
    '''
    all_link_lst          = []
    work_life_balance_lst = []
    pay_benefit_lst       = []
    job_secured_lst       = []
    management_lst        = []
    culture_lst           = []

    all_tail_string = ['?lang=en'] + ['?start=' + str(i) + '&lang=en' for i in list(np.arange(1,500)*20)]
    all_link_lst = ['https://www.indeed.com/cmp/' + company + '/reviews' + i for i in all_tail_string]

    topic_tail_string = ['&lang=en'] + ['&start=' + str(i) + '&lang=en' for i in list(np.arange(1,250))]

    work_life_balance_lst = ['https://www.indeed.com/cmp/' + company + '/reviews?ftopic=wlbalance' + i for i in topic_tail_string]

    pay_benefit_lst = ['https://www.indeed.com/cmp/' + company + '/reviews?ftopic=paybenefits' + i for i in topic_tail_string]

    job_secured_lst = ['https://www.indeed.com/cmp/' + company + '/reviews?ftopic=jobsecadv' + i for i in topic_tail_string]

    management_lst = ['https://www.indeed.com/cmp/' + company + '/reviews?ftopic=mgmt' + i for i in topic_tail_string]

    culture_lst = ['https://www.indeed.com/cmp/' + company + '/reviews?ftopic=culture' + i for i in topic_tail_string]

    #num_str_list = ['lang=en'] + ['?start=' + str(i) + '&lang=en' for i in list(np.arange(1,200)*20)]
    return all_link_lst, work_life_balance_lst, pay_benefit_lst, job_secured_lst, management_lst, culture_lst



def get_user_id(browser):
    user_id = browser.find_elements_by_css_selector('div.cmp-review-container')
    list_user_id = []
    for id in user_id[1:]:
        list_user_id.append(id.get_attribute('id'))
    return list_user_id

def get_title(browser):
    lst_titles = []
    cmp_content = browser.find_element_by_css_selector('div#cmp-content')
    review_titles = cmp_content.find_elements_by_css_selector('div.cmp-review-title')
    for title in review_titles[1:]:
        lst_titles.append(title.text)
    return lst_titles

def get_job_title(browser):
    job_titles_lst = []
    com_content = browser.find_element_by_css_selector('div#cmp-content')
    job_titles = com_content.find_elements_by_css_selector('span.cmp-reviewer-job-title')
    for job in job_titles[1:]:
        job_titles_lst.append(job.text)
    return job_titles_lst


def get_job_location(browser):
    loc_lst = []
    com_content = browser.find_element_by_css_selector('div#cmp-content')
    locations = com_content.find_elements_by_css_selector("span.cmp-reviewer-job-location")
    for loc in locations[1:]:
        loc_lst.append(loc.text)
    return loc_lst

def get_dates(browser):
    date_lst = []
    com_content = browser.find_element_by_css_selector('div#cmp-content')
    dates = com_content.find_elements_by_css_selector("span.cmp-review-date-created")
    for date in dates[1:]:
        date_lst.append(date.text)
    return date_lst

def get_text_reviews(browser):
    main_text_reviews = []
    com_content = browser.find_element_by_css_selector('div#cmp-content')
    reviews = com_content.find_elements_by_css_selector("span.cmp-review-text")
    for review in reviews[1:]:
        main_text_reviews.append(review.text)
    return main_text_reviews

def get_reviews_score(browser):
    overalls, balances, benefits, security, management, culture = [], [], [], [], [], []
    com_content = browser.find_element_by_css_selector('div#cmp-content')
    reviews_score = com_content.find_elements_by_css_selector('span.cmp-Rating-on')[6:]
    for index, score in enumerate(reviews_score):
        if index%6==0:
            overalls.append(score.get_attribute('style'))
        elif index%6==1:
            balances.append(score.get_attribute('style'))
        elif index%6==2:
            benefits.append(score.get_attribute('style'))
        elif index%6==3:
            security.append(score.get_attribute('style'))
        elif index%6==4:
            management.append(score.get_attribute('style'))
        elif index%6==5:
            culture.append(score.get_attribute('style'))
    return overalls, balances, benefits, security, management, culture

def convert_to_dataframe(company_dict):
    len_lst = []
    data_dict = OrderedDict(company_dict[list(company_dict.keys())[0]])
    for k, v in data_dict.items():
        len_lst.append(len(v))
    min_len = min(len_lst)
    for k, v in data_dict.items():
        data_dict[k] = v[:min_len]
    df = pd.DataFrame(data_dict)
    df['company_name'] = list(company_dict.keys())[0]
    return df

def export_data_to_csv(df, company, topic):
    file_name = '/Users/hatran/project/galvanize/capstone/data/' + company + '_' + topic + '_data.csv'
    new_name = company + '_' + topic + '_data.csv'
    df.to_csv(file_name)


def upload_data(file_name, new_name):
    s3 = boto3.resource('s3')
    bucket_name = "capstone_raw_data_ha_galvanize"
    try:
        s3.meta.client.upload_file(file_name, bucket_name, new_name)

    except Exception as e:
        print(e)  # Note: BucketAlreadyOwnedByYou means you already created the bucket.
