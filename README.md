# Employee Reviews: What is new?

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


![Alt text](images/predictive_models.png?raw=true "Title")



With the conclusion in stage 1, I designed and built a new scoring system in which the final score is a function of rating and text comments. The text comments were processed by natural language processing sentiment analysis. With this new scoring system, each company will have its own new work-life balance score, compensation-benefit score, job security score, management score, and culture score. The top 1 companies that were:

![Alt text](images/top_1_companies.png?raw=true "Title")

Utilizing the new text comments - rating score system, a recommender system was developed using the matrix factorization method. To improve the recommendation capacities, there were two modifications: 1. both text-comment and rating were used in generating the feature matrix, 2. personal, instutional, temporal factors were incorperated into missing imputation to solve the "cold start" problem. A website has been developed with which, each job seeker can provide their own weights on five features of work-life balance, compensation-benefit, job security, management, and culture and the website will provide the list of companies that are best fit for him.

Have a lovely day!














