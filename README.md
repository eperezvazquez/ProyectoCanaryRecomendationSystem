## Project Overview
This project was developed using data from two different databases: one relational SQL database, SIGES, and the other non-relational, MongoDB. Both were converted to SQL using Pentaho from PDI, with SIGES containing over seven years of data. The project is designed to leverage these extensive datasets to predict whether AGESIC projects from 2011 to the present will complete within, over, or under their budget, using various predictive variables obtained through the SIGES system.

### Why This Project?
The AI applied here aims to increase project management efficiency, improve budget handling, and deliver value that positively impacts public administration and services.

#### Assumptions

* The analysis is objective and strictly adheres to organizational guidelines. 
* Projects are flagged for budget deviations above 20% (red) and 10% (yellow) based on preset thresholds. 
* Time deviations, although a standard guideline, are adjustable per project and are not typically considered. 
* Budget deviations are assessed according to organizational standards.

#### Detailed EDA

We conducted a detailed Exploratory Data Analysis (EDA) on 362 columns and 183,281 records. This was segmented into subsets for deeper exploration:

* Risk Analysis: Analysis of 15 variables across 71,383 risk entries showed key findings such as risk concentration, significant gaps in risk identification, and an often oversimplified risk assessment approach.

* Lessons Learned: Highlighted the underutilization of documented lessons that could improve future project management practices.

* Program Analysis: Analysis of 24 columns showed that 98.15% of programs are actively contributing to projects.

* Other Project Data: Strong correlations were found between project deliverables, schedules, budgets, risks, and lessons learned.

* Deliverables: Focused on the structure of project deliverables and documentation, revealing the need for better management.

* Project Scheduling: Examined practices around project timelines, noting a lack of traditional “traffic-light” indicators and emphasizing actual start and end dates.

* Budget Analysis: Investigated budget records and discrepancies in the use of budgeting versus payment currencies.

* Specific Regression Model Analysis
We applied logistic regression to predict project challenges based on identified variables.

* Payment Time Series
Analyzed payment behaviors and the impact of currency fluctuations on financial planning.

* Recommendation System Analysis
Evaluated user feedback and system performance to refine customer satisfaction and service delivery.

#### Applied Models and Insights

* Classification and Regression Analysis: Applied logistic regression to model and predict project outcomes related to time, budget, and scope risks.

* Time Series Forecasting: Used the Prophet model to forecast budget trends, taking into account seasonal variations and external factors such as holidays.

* Neural Networks: Applied deep learning models like RNNs, LSTMs, and GRUs to capture complex data patterns beyond the reach of traditional models.

* Recommendation Systems: Deployed various algorithms, including Singular Value Decomposition (SVD), which performed well in predicting project and resource allocations.

#### Strategic Recommendations

* Risk Management Enhancement: Improve documentation, address data gaps, and refine risk assessment processes.

* Lessons Learned: Increase the capture and application of lessons learned to improve future project outcomes.

* Budget and Financial Controls: Standardize currency usage and review financial controls to improve budget accuracy.

* Technological Advancements: Further integrate advanced AI and machine learning models to enhance predictive accuracy and operational efficiency.

#### Future Directions

* Neural Network Implementation: Plan to deploy bidirectional neural networks and explore random forest models to enhance predictive accuracy.

* Recommendation System Development: Aim to implement a comprehensive ranking system within the existing recommendation framework to optimize resource allocation and project delivery.

URL: https://appregresionusc-c67e8d36ff42.herokuapp.com/
email_heroku: elisa.perez.vazquez@rai.usc.gal
emial_github: personal.
