# Employee Reviews: What is New?

![Alt text](images/Wordcount.png?raw=true "Title")

## Confidentiality
The analysis data was published on indeed.com. Company and employee information was truncated in the "publishable folder". If you need to discuss on the raw data access, please contact me at hatran84{@}outlook.com

## Project Introduction
As more than 70% of job seekers view company reviews on employee rating sites before applying and making a career decision, these employee ratings become tremendously important. While the overall ratings of these employee rating sites are clear, they can be misleading due to the discrepancy between text and rating scores.

In this project, there are three stages:
    Stage 1 - hypothesis testing: determined if the text comments add further information into employee   satisfaction.
    Stage 2 - back-end product development: constructed a new rating scale by combining text and classical ratings.
    Stage 3 - front-end product development: developed a recommender system for each job seeker.
  
![Alt text](images/Project_introduction.png?raw=true "Title")

In my stage 1, I used the information of employees staying/leaving companies as the imperfect surrogate information for employee satisfaction. This information was also used as the predictor for two nested models, model without text comments and model without text comments. Persistently, models without text comments performed better in predicting the surrogate information of employee satisfaction:

## Data Understanding
![Alt text](images/Data_understanding.png?raw=true "Title")

More than 60,000 reviews of 33 companies that have offices in Seattle area and recruit data scientist/data analyst positions were scrapped from Indeed.com using the Selenium library. 

## Modeling and Results
### First stage
In the first stage, the information of employees leaving/staying companies was used as the surrogate information for employee satisfaction. Two nested models - model including text reviews and model excluding text reviews were compared by using logistic regression, decision tree, adaboosting, naïve bayes, random forest, gradient boosting methods. Text reviews were clustered by the KMeans algorithm. 

The model with text reviews persistently performed better in both log loss and AUC (area under curve) scores. Gradient Boosting models had the best performances with 76% of accuracy. As a result, text comments were concluded to provide further information on employee satisfaction in addition to the ratings and should be included in the overall employee rating scores. 

![Alt text](images/predictive_models.png?raw=true "Title")


### Second stage
In the second stage, a new company rating system was constructed as the function of text reviews and classical ratings. The text reviews were processed by Natural Language Processing sentiment analysis to produce "text" scores. 

Using the new developed scoring system, each company possessed a new set of scores such as work-life balance, benefit and compensation, job security & advancement, management, and culture score. Saleforces inc. was rated highest by its current and former employees in the culture scale. 

![Alt text](images/top_1_companies.png?raw=true "Title")

### Third stage

To improve the recommendation capacities, both text comments and ratings were incorporated in generating the feature matrix. The personal, institutional, temporal factors were also integrated in missing imputation. A website has been launched to provide recommendations for job seekers based on their personal desires. 


![Alt text](images/Matrix_factorization.png?raw=true "Title")


## Have a lovely day!














