# diabetes-prediction

![diabetes](https://github.com/dspataru/diabetes-prediction/assets/61765352/5b2eb604-ef51-4f85-b050-136fedfd6756)


## Table of Contents
* [Introduction](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#introduction)
* [Background](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#background)
* [Data Source](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#data-source)
* [Preparing the Data for the Models](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#preparing-the-data-for-the-models)
* [Model Description](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#model-description)
* [Analysis and Results](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#analysis-and-results)
* [Conclusion](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#conclusion)
* [Future Work](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#future-work)
* [References](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#references)

## Introduction

In recent years, the intersection of healthcare and machine learning has emerged as a transformative field, offering innovative solutions to longstanding challenges. Among the myriad applications, predicting and managing chronic diseases, such as diabetes, has become a focal point. Complications like heart disease, vision loss, lower-limb amputation, and kidney disease are associated with chronically high levels of sugar remaining in the bloodstream for those with diabetes. While there is no cure for diabetes, strategies like losing weight, eating healthily, being active, and receiving medical treatments can mitigate the harms of this disease in many patients. Early diagnosis of diabetes can lead to lifestyle changes and more effective treatment, making predictive models for diabetes risk important tools for public and public health officials.

The integration of machine learning models in healthcare has played a pivotal role in advancing diagnostic and predictive capabilities. Among these models, classification algorithms can be a useful tool for predicting the onset of diabetes. This report delves into the significance of employing machine learning models, particularly classification models, in predicting diabetes. By exploring the differences between prominent classification models such as K-nearest neighbors, neural networks, and random forests, this report aims to find a model that is most accurate at predicting diabetes.

Understanding the intricacies of these models is crucial, as they leverage distinct methodologies to analyze patient data and make predictions. K-nearest neighbors relies on proximity-based relationships, neural networks emulate the complex structure of the human brain, and random forests employ ensemble learning to enhance predictive accuracy. Examining these models in tandem allows for a nuanced appreciation of their individual contributions to diabetes prediction.

An essential facet of utilizing classification models in healthcare is the assessment of their performance. This report will explore various metrics and techniques employed to evaluate the efficacy of these models, ensuring that the chosen algorithms meet the stringent requirements of reliability and accuracy demanded by the healthcare domain. Furthermore, optimization strategies will be discussed, elucidating ways to fine-tune these models for enhanced performance in real-world scenarios.

In summary, diabetes affects the health of millions of people and puts an enormous financial burden on the US economy. This exploration aims to develop predictive models to identify risk factors for diabetes which could help facilitate early diagnosis and intervention and also reduce medical costs.

#### Libraries used
psycopg2, sqlalchemy, pandas, numpy, sklearn, tensorflow, matplotlib, seaborn

## Background

## Data Source

Data for this project is taken from the 2022 data off the CDCs website from their Behavioral Risk Factor Surveillance System sector (BRFSS). The BRFSS is the United States’s premier system of health-related telephone surveys that collect state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. They complete more than 400,000 adult interviews each year and keep a record of all of the survey data and documentation, including the questionnaries readily available on their website: [CDC BRFSS](https://www.cdc.gov/brfss/annual_data/annual_2022.html).

The [BRFSS Overview](https://www.cdc.gov/brfss/annual_data/2022/pdf/Overview_2022-508.pdf) document provides context for what was in the 2022 question set and descriptions of their samples including the target population, geographic populations of interest, and more. They also describe their interviewing procedure for data collection, and how they processed their data. The BRFSS prepare codebooks that contain the variable names for each column in the dataset, which column number corressponds to that variable, what question was asked for that column, and the the values are, along with much more information in a table format. Below is an example of what one of the tables in the codebook looks like:

![BRFSS codebook sample](https://github.com/dspataru/diabetes-prediction/assets/61765352/afa5e99f-6a84-4c2d-a4a4-4e7c606eb966)

These tables were used to extract information directly related to diabetes. The dataset originally has 326 features (columns) and 445,132 records for 2022, but based on diabetes disease research regarding factors influencing diabetes disease and other chronic health conditions, only select features are included in this analysis. As seen in the table above, there is a "Section Name", and a set of questions belongs to each section name. We searched for "Diabetes" related questions and came across a the following questions:
1. What type of diabetes do you have?
2. Are you now taking insulin?
3. About how many times in the past 12 months has a doctor, nurse, or other health professional checked you for A-one-C?
4. When was the last time you had an eye exam in which the pupils were dilated, making you temporarily sensitive to bright light?
5. When was the last time a doctor, nurse or other health professional took a photo of the back of your eye with a specialized camera?
6. When was the last time you took a course or class in how to manage your diabetes yourself?
7. Have you ever had any sores or irritations on your feet that took more than four weeks to heal?
8. When was the last time you had a blood test for high blood sugar or diabetes by a doctor, nurse, or other health professional?
9. Has a doctor or other health professional ever told you that you had prediabetes or borderline diabetes?  (If “Yes” and respondent is female, ask: “Was this only when you were pregnant?”)

These were questions that the BRFSS asked over the phone related specifically to diabetes and pre-diabetes. In addition to the above questions, diabetes research has found the there are other important risk factors to be taken into consideration, including blood pressure, cholesterol, smoking, obesity, age, sex, race, diet, exercise, alcohol consumption, SMI, household income, sleep, frequency of doctor visits, medical care coverage, and mental and physical health. Unfortunately, the 2022 questionnaire does not include questions relation to blood pressure, cholesterol, or diet, however, there are many other questions that were used to provide data for the predictive models.


## Preparing the Data for the Models

To predict if an individual has diabetes, a subset of the cleaned data was used and transformed into a new dataset to be fed into the models. The following were the columns of interest from the raw data:
1. 'DISPCODE': This column contains two values 1100 for completed interviews, and 1200 for incomplete interviews. For the purposes of this project, we only want to include data from complete interviews. In the data cleaning process, we drop the rows where DISPCODE == 1200 which reduces the data from 445,132 entries, to 353,271 entries.
2. 'DIABETE4': The question that was asked: (Ever told) (you had) diabetes? (If ´Yes´ and respondent is female, ask ´Was this only when you were pregnant?´. If Respondent says pre-diabetes or borderline diabetes, use response code 4.) In this dataset 1=Yes, 2=Yes but told only during pregnancy, 3= No, 4=No, prediabetes or borderline, 7=Don't know/not sure, and 9=Refused.
3. 'PDIABTS1': The question that was asked: When was the last time you had a blood test for high blood sugar or diabetes by a doctor, nurse, or other health professional? This is a value between 1-9 where 1 is 'Within the past year (anytime less than 12 months ago)', 8 is never, and 9 is refused.
4. 'CHKHEMO3': The questions that was asked:  About how many times in the past 12 months has a doctor, nurse, or other health professional checked you for A-one-C? This is a value between 1-76 for people that responded. A value of 88=None, 98=Never heard of it, 77=Don't know/not sure, and 99=Refused.
5. '_BMI5CAT': This is a calculated value from other columns that categories individuals into one of four categories of Body Mass Index (BMI). 1=underweight, 2=normal weight, 3=overweight, and 4=obese.
6.  '_SMOKER3': This column contains buckets individuals into four levels of smoker status: 1=Everyday smoker, 2=Someday smoker, 3=Former smoker, 4=Non-smoker.
7.  'EYEEXAM1': Question that was asked: When was the last time you had an eye exam in which the pupils were dilated, making you temporarily sensitive to bright light? 1=Within the past month, 2=Within the past year, 3=Within the past two years, 4=two or more years ago, 7=don't know/not sure, 8=never, 9=refused.
8.  'DIABEYE1': Question that was asked: When was the last time a doctor, nurse or other health professional took a photo of the back of your eye with a specialized camera? The values in this column are the same as the above question for 'EYEEXAM1'.
9.  'CVDSTRK3': Question that was asked: (Ever told) (you had) a stroke. 1=Yes, 2=no, 7=Don't know/not sure, and 9=Refused. This question was related to chronic health conditions.
10.  '_MICHD': Question that was asked: Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI). 1=Reported having MI or CHD, 2=Did not report having MI or CHD. This question was related to chronic health conditions.
11.  '_TOTINDA': Question that was asked: Adults who reported doing physical activity or exercise during the past 30 days other than their regular job. 1=Had physical activity or exercise, 2=No physical activity or exercise in last 30 days, 3=Don’t know/Refused/Missing.
12.  '_RFDRHV8':  Question that was asked: Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week). 1=No, 2=Yes, 9=Don’t know/Refused/Missing.
13.  '_HLTHPLN': Adults who had some form of health insurance. 1=Have some form of insurance, 2=Do not have some form of health insurance, 9=Don´t know, refused or missing insurance response.
14.  'MEDCOST1': Question asked: Was there a time in the past 12 months when you needed to see a doctor but could not because you could not afford it? 1=Yes, 2=No, 7=don't know/not sure, 9=refused.
15.  'GENHLTH', 'MENTHLTH', 'PHYSHLTH', 'DIFFWALK': These questions are related to general health, mental health, physical health, and walking frequency.
16.  'SEXVAR', '_AGEG5YR', '_EDUCAG', 'INCOME3': The questions asked for these columns are related to demographics.

The information in the dataframe was examined using the ```.info()``` method to review the non-null count in each column. The output is seen below:

![raw_data.info()](https://github.com/dspataru/diabetes-prediction/assets/61765352/8950547f-88af-4780-bcdc-8a4dade08998)

The CHKHEMO3, EYEEXAM1, DIABEYE1, and PDIABTS1 columns are missing many values. As a result, we drop these columns for the first attempt at creating a dataset to input to the ML models. Following this, the ```.dropna()``` method is used to drop all of the row entries with NaN values. The resulting dataset contains 17 columns (features) and 326,519 rows. The next step is to modify and clean the values to be more suitable to the ML algorithms. In order to be able to do this part, each column in the dataset was reviewed against the codebook which says what each column is, and what the values in each column correspond to. A breakdown of what was down can be found in the [data_model_cleaning.ipynb]() jupyter notebook in section 2.2. Finally, the columns were renamed to be more understandable. The resulting dataframe contained 252,888 rows, as some rows were dropped in the cleaning process, with 17 features, and the following classes for the diabetes column (target column):
* No diabetes (0): 211,801 observations.
* Pre-diabetes or borderline diabetes (1): 5893 observations.
* Yes diabetes (2): 35,194 observations.

This dataset was uploaded to an postgres database hosted by AWS RDS to be easily accessable by the whole team.

## Feature Selection

Feature selection is a crucial step in the process of building machine learning models as it plays a pivotal role in enhancing model performance and interpretability. The significance of feature selection lies in its ability to improve the model's efficiency by focusing on the most relevant and informative features while discarding irrelevant or redundant ones. By reducing the dimensionality of the dataset, feature selection helps mitigate the curse of dimensionality, which can adversely affect model training time and generalization to unseen data. For this project, we view the correlation matrix to check the correlation between features.


## The Models

### Model Description

Various models were used to predict diabetes using a subset of the master data, including K-Nearest Neighbour, Random Forest, and Deep Learning model. These models were imported from the ```sklearn``` library. The data was split into a training and testing set, using the Diabetes column as the target variable, and the features as selected during the feature selection process described in the previous section.

### Dealing with Class Imbalance

The raw data taken from the CDC's website contained four categories as described in data sections above. When cleaned and grouped, no diabetes had almost 5 times more observations than the diabetes, gestantional diabetes, and pre-diabetes/borderline diabetes combined. This imbalance in the dataset can lead to models that are biased towards the majority class and have poor generalization, resulting in poor precision and recall for the minority class. This is another reason why the accuracy of a model should not be the only metric that is evaluated when determining the performance of a model. Imbalanced classes can provide misleading evaluation metrics as they can produce a high accuracy, but this is a result of the accuracy of the majority class prediction and not representative of the minority class prediction accuracy.

As a result, we deal with the class imbalance by using resampling techniques including both oversampling of the minority class, and undersampling of the majority class. In addition, we use ensemble methods such as Random Forests in this exploration that can handle imbalanced datasets more effectivetly than individual models.For this project, we used the following methods of dealing with class imbalance:
1. RandomOversampler: This randomly duplicates instances from the minority class (or classes) until a more balanced distribution is achieved. From imbalanced-learn library.
2. Random Undersampling: This does the oppositre of the above. Instead of oversampling the minority class, we undersample the majority class.
3. Synthetic Minority Oversampling Technique (SMOTE): SMOTE is specifically designed to tackle imbalanced datasets by generating synthetic samples for the minority class. This algorithm can be found in the imblearn library.
4. Adaptive Synthetic Sampling Approach (ADASYN): ADASYN is a generalized form of the SMOTE algorithm. This algorithm also aims to oversample the minority class by generating synthetic instances for it. But the difference here is it considers the density distribution which decides the number of synthetic instances generated for samples which difficult to learn. This algorithm can be found in the imblearn library.

### Model Evaluation

Several metrics were used to evaluate each model, including:
1. Confusion matrix: A confusion matrix is a table that is used to evaluate the performance of a classification algorithm on a set of test data for which the true values are known. It provides a summary of prediction results and reveals insights into the model's behavior.
```
                Predicted Negative    Predicted Positive
Actual Negative        TN                   FP
Actual Positive        FN                   TP
```
  * True Positive (TP): The model correctly predicted positive instances.
  * True Negative (TN): The model correctly predicted negative instances.
  * False Positive (FP): The model incorrectly predicted positive instances.
  * False Negative (FN): The model incorrectly predicted negative instances.
2. Classification report: A classification report is a table that provides a comprehensive evaluation of the performance of a classification model. The report includes various metrics that help assess the quality of predictions made by the model. Common metrics in a classification report include precision, recall, F1-score, and support.
  * Precision is the ratio of true positive predictions to the total number of predicted positives (true positives + false positives). It measures the accuracy of positive predictions and is also known as the Positive Predictive Value (PPV). A high precision indicates a low false positive rate. $Precision = (True Positives)/(True Positives + False Positives)$
  * Recall is the ratio of true positive predictions to the total number of actual positives (true positives + false negatives). It measures the ability of the model to capture all the relevant instances of the positive class. A high recall indicates a low false negative rate. $Recall = (True Positives)/(True Positives + False Negatives)$
  * The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall, considering both false positives and false negatives. The F1-score is particularly useful when there is an uneven class distribution. It is calculated by: $F1-Score = 2*((Precision x Recall)/(Precision + Recall))$.
  * Support is the number of actual occurrences of the class in the specified dataset. It is the count of true instances for each class. Support helps interpret the significance of precision and recall, especially when dealing with imbalanced datasets.
  * Accuracy is the ratio of correctly predicted instances to the total number of instances. Accuracy alone might be misleading in imbalanced datasets, so it's often important to consider it along with precision, recall, and the F1-score.
3. ROC curve: An ROC (Receiver Operating Characteristic) curve is a graphical representation of the performance of a binary classification model at various classification thresholds. It illustrates the trade-off between the true positive rate (sensitivity or recall) and the false positive rate (1 - specificity) as the decision threshold for classifying positive instances is varied. Interpretation of an ROC curve:
  * The ROC curve provides a visual representation of how well a binary classifier is able to distinguish between the two classes.
  * A curve that hugs the upper-left corner of the plot indicates better performance, as it corresponds to higher true positive rates and lower false positive rates across different threshold values.
  * The diagonal line (from the bottom-left to the top-right) represents the performance of a random classifier.

## Analysis and Results

### Analysis with Dataset 1

The first dataset that was used contained the following columns: 'GENDER', 'INCOME', 'WEIGHT', 'BMI', 'RACE', 'AGE', 'DIABETES', 'PHYSHLTH', 'MENTHLTH', 'EXERCISE', 'HLT_INSURANCE', 'PERSONAL_DOC', 'CHECKUP1', 'HRT_ATTACK', 'HRT_DISEASE', 'STROKE', 'ARTHRITIS', '_SMOKER3', 'DIFFWALK', 'EDUCATION', 'HEIGHT'.

Several classifier models were run with different resampling techniques and below is a summary of the evaluation metrics for each model:

![dataset1-models-comparison](https://github.com/dspataru/diabetes-prediction/assets/61765352/e0d95308-d221-4b4e-b00e-c10a6a741301)


![F1-Score](https://github.com/dspataru/diabetes-prediction/assets/61765352/860e1b6d-5734-4138-9867-151cc4e551cb)

![Recall](https://github.com/dspataru/diabetes-prediction/assets/61765352/c8cb062d-eae5-4112-9ad2-a3b78c5dcb21)

![Accuracy](https://github.com/dspataru/diabetes-prediction/assets/61765352/0d505005-cca1-4e1f-bdae-aeac6f2ea343)

### Analysis with Dataset 2


# Feature Analysis of Data set 2

1. Logistic Regression and  Random Rainforest

As per the picture provided, it is evident that the likelihood of having diabetes or not having diabetes is directly related to the age of the person. Approximately 25% of the dataset is covered by age, followed by income and general health. This insight emphasizes the significance of age in understanding and predicting diabetes in the given dataset.
￼
![Pie Chart](https://github.com/dspataru/diabetes-prediction/assets/136105558/6a2b555a-0d26-452a-b632-8133d3960bde)

![Features Importances](https://github.com/dspataru/diabetes-prediction/assets/136105558/581baa07-e32f-4370-aad8-d332d7686632)

# Model Description of Data set 2

1. Logistic Regression and  Random Rainforest

This project involves the application of machine learning techniques to a dataset. The primary objective is to perform classification using logistic regression and a random forest algorithm. Before implementing these algorithms, the dataset undergoes a cleaning and preprocessing phase, followed by a strategic filtering process. Subsequently, the data is divided into equal parts and subjected to both under-sampling and over-sampling techniques to address potential class imbalance issues

    Steps

    1. Data Cleaning and Preprocessing
    - The initial step in this project is to clean and preprocess the raw dataset. This involves handling missing values, removing duplicates, and converting data types if necessary. The cleaned dataset serves as the foundation for subsequent analyses.

    2. Filtering the Dataset
    - To streamline the dataset for machine learning, a filtering process is applied. This involves selecting relevant features and eliminating unnecessary ones. The goal is to create a more focused dataset that optimizes the performance of the machine learning algorithms.

    3. Data Division
    - The filtered dataset is then divided into equal parts. This step is crucial for training and evaluating machine learning models. The division ensures that the models have sufficient data for both learning and testing, promoting generalizability.

    4. Addressing Class Imbalance
    - Class imbalance can significantly impact the performance of a machine learning model. To mitigate this issue, the dataset undergoes both under-sampling and over-sampling. Under-sampling involves reducing the size of the majority class, while over-sampling involves increasing the size of the minority class. This creates a balanced dataset that improves the model's ability to accurately predict both classes.

    5. Machine Learning Algorithms
    - Two machine learning algorithms, logistic regression and random forest, are employed for classification. Logistic regression is a linear model suitable for binary classification tasks, while the random forest algorithm is an ensemble method that combines multiple decision trees for improved accuracy and robustness. 

    6. Evaluation
    - The classification performance of the models is assessed using a classification report. This report provides metrics such as precision, recall, and F1-score for each class, offering a comprehensive understanding of the models' predictive capabilities. Additionally, accuracy is calculated to measure the overall correctness of the models.

# Analysis of Logistic Regression and Random Rainforest of Data set 2

The dynamic duo of logistic regression and random forests. Imagine logistic regression as a savvy detective making decisions based on clues—it predicts outcomes, like whether someone has a diabetes or not. Now, pair that with random forests—it's like having a team of detectives, each with a unique perspective, working together to crack the case. The combination is powerful! Logistic regression sets the stage, and random forests bring diversity to the decision-making process, enhancing accuracy. Together, they're our tech detectives, for making sense of complex data and delivering reliable predictions.

In this example, we first load the diabetes dataset using pandas library. Then we split the data into training and testing sets using train_test_split function. Next, we create a logistic regression model using LogisticRegression class and train the model on the training data using fit method. Finally, we make predictions on the testing data using predict method and evaluate the model's accuracy using accuracy_score function.

1. Equal Dataset

     1.1 Classification Report

![Screenshot 2023-11-23 at 10 09 08 PM (2)](https://github.com/dspataru/diabetes-prediction/assets/136105558/6ea8f38e-451e-4017-a394-de822688d39e)

     1.2 ROC Chart:

![Receiver Operating Characteristic (ROC) Curve](https://github.com/dspataru/diabetes-prediction/assets/136105558/f590f7b4-275a-4c7a-8f58-4585c90058b3)

 2. Over Sampling and Under Sampling:

   2.1 Classification Report:

![Screenshot 2023-11-23 at 10 07 22 PM (2)](https://github.com/dspataru/diabetes-prediction/assets/136105558/edeea60f-fea4-488e-96b4-924294384056)
 
   2.2 ROC Chart:

![over and under ROC](https://github.com/dspataru/diabetes-prediction/assets/136105558/6157b52d-a442-4a79-9a0c-0169e94752ae)

### Analysis with Dataset 3


### Analysis with Dataset 4:
In this section we have tried to build a model that can successfully categories the diabetes types. Type 1 and type 2. In order to do that we have taken into consideration into several machine learning models such as logistic regression, random forest, random decision classifier, K-nn neighbors, XGBoost, SVM. We also tried to find out the Neural Network model in order to interpret the categories. We will learn with the given dataset which model successfully predict diabetes types.

To set up the environment we have first install the psycopg2 module. We had to create a connection to the database. In order to do that, we had to run the below code. Basically we needed to connect to the table called claean_diab_info which is the table used to train and predict the categories of the diabetes.

import psycopg2
from sqlalchemy import create_engine
import pandas as pd

hostname = 'diabetes-dataset.cwpas6tssjkb.us-east-1.rds.amazonaws.com'
database = 'diabetes_database'
username = '' # enter your username manually
password = '' # enter your password manually
port_id = 5432

try:
    conn = psycopg2.connect(
        host = 'diabetes-dataset.cwpas6tssjkb.us-east-1.rds.amazonaws.com',
        dbname = database,
        user = 'mislam',
        password = 'MPIA-dd#',
        port = 5432)
    print("Connected to the database!")

except Exception as e:
    print(f"Unable to connect to the database. Error: {e}")

# example query to grab all of the columns
sql_query = "SELECT * FROM clean_diab_info"
df = pd.read_sql_query(sql_query, conn)
df.head()


 

We had created a new dataframe diabetes_df which consisted of the below columns:

"DIABETES","DIABTYPE","AGE","INSULIN_Y/N","A-one-C_test","EYEEXAM1","DIABEYE1","DIAB_MNGMT","PERSONAL_DOC","HRT_DISEASE","STROKE","ARTHRITIS"



In order to train and test the data we had to split the data into two different parts> one training dataset and the test dataset

X=diabetes_category[["DIABETES","AGE","INSULIN_Y/N","A-one-C_test","EYEEXAM1","DIABEYE1","DIAB_MNGMT","PERSONAL_DOC","HRT_DISEASE","STROKE","ARTHRITIS"]]

y=diabetes_category["DIABTYPE"]


The first model that we tried is the logistic regression. The results are following:


Accuracy: 0.9072681704260651
Classification Report:
              precision    recall  f1-score   support

         1.0       0.80      0.04      0.08       191
         2.0       0.91      1.00      0.95      1804

    accuracy                           0.91      1995
   macro avg       0.85      0.52      0.52      1995
weighted avg       0.90      0.91      0.87      1995



 


We can see that the model is not able to predict the type 1 diabetes and having a f1-score of .08. The next model that we tried 


Accuracy: 0.8426065162907268
Classification Report:
              precision    recall  f1-score   support

         1.0       0.20      0.22      0.21       191
         2.0       0.92      0.91      0.91      1804

    accuracy                           0.84      1995
   macro avg       0.56      0.56      0.56      1995
weighted avg       0.85      0.84      0.85      1995






 


This is one of the best models to predict both correctly but it has overall accuracy low as 84%.


Then we tried the random forest classifier model and the below is the accuracy is below:

 

Accuracy: 0.8982456140350877
Classification Report:
              precision    recall  f1-score   support

         1.0       0.41      0.14      0.20       191
         2.0       0.91      0.98      0.95      1804

    accuracy                           0.90      1995
   macro avg       0.66      0.56      0.57      1995
weighted avg       0.87      0.90      0.87      1995



We see that it is one of the best models to predict having the highest f1-scorre 20% and overall accuracy as 90%


We also drew the importance matrix from here:

 
By taking only the important columns we increased the f1-score a little bit of 22% but the overall remains the same.

 


Then we took into account the SV technique. And the following is the result:

Accuracy: 0.9042606516290727
Classification Report:
              precision    recall  f1-score   support

         1.0       0.00      0.00      0.00       191
         2.0       0.90      1.00      0.95      1804

    accuracy                           0.90      1995
   macro avg       0.45      0.50      0.47      1995
weighted avg       0.82      0.90      0.86      1995



However, it fails to draw the type 1 



Then the next model that we tried is the Knn classifier.

 


Accuracy: 0.8962406015037594
Classification Report:
              precision    recall  f1-score   support

         1.0       0.37      0.12      0.18       191
         2.0       0.91      0.98      0.94      1804

    accuracy                           0.90      1995
   macro avg       0.64      0.55      0.56      1995
weighted avg       0.86      0.90      0.87      1995



This is also a good model to predict


The next model we tried is the XGB classifier.

 


Accuracy: 0.9027568922305764
Classification Report:
              precision    recall  f1-score   support

           0       0.47      0.13      0.20       191
           1       0.91      0.98      0.95      1804

    accuracy                           0.90      1995
   macro avg       0.69      0.56      0.58      1995
weighted avg       0.87      0.90      0.88      1995


It is the second best model so far with the f1 score 20 and accuracy 90%.


We also tried to tune up the NN however the accuracy is bit lower than the usual models. See below:


 


Trial 60 Complete [00h 00m 16s]
val_accuracy: 0.08540496975183487

Best val_accuracy So Far: 0.08540496975183487
Total elapsed time: 00h 06m 38s


Then we have tried to scaled the dataset and tired to do the same models. 
Logistic regression=

Accuracy: 0.9157979149959904
Classification Report:
              precision    recall  f1-score   support

         1.0       1.00      0.01      0.03       213
         2.0       0.92      1.00      0.96      2281

    accuracy                           0.92      2494
   macro avg       0.96      0.51      0.49      2494
weighted avg       0.92      0.92      0.88      2494





Decision tree=

Accuracy: 0.4843624699278268
Classification Report:
              precision    recall  f1-score   support

         1.0       0.10      0.61      0.17       213
         2.0       0.93      0.47      0.63      2281

    accuracy                           0.48      2494
   macro avg       0.51      0.54      0.40      2494
weighted avg       0.86      0.48      0.59      2494




random_forest:

Accuracy: 0.8376102646351243
Classification Report:
              precision    recall  f1-score   support

         1.0       0.10      0.11      0.11       213
         2.0       0.92      0.91      0.91      2281

    accuracy                           0.84      2494
   macro avg       0.51      0.51      0.51      2494
weighted avg       0.85      0.84      0.84      2494



Xgb_classifier=

Accuracy: 0.899749373433584
Classification Report:
              precision    recall  f1-score   support

           0       0.42      0.12      0.19       191
           1       0.91      0.98      0.95      1804

    accuracy                           0.90      1995
   macro avg       0.67      0.55      0.57      1995
weighted avg       0.87      0.90      0.87      1995


knn_classifier_1:

Accuracy: 0.8927318295739348
Classification Report:
              precision    recall  f1-score   support

         1.0       0.33      0.12      0.17       191
         2.0       0.91      0.98      0.94      1804

    accuracy                           0.89      1995
   macro avg       0.62      0.55      0.56      1995
weighted avg       0.86      0.89      0.87      1995




As type 1 diabetes is genetical and it does not depend upon anything As a result, it is quite hard to predict type 1 diabetes


## Conclusion

Below is a summary of the results of the best model performance given the different datasets.
![summary-of-results](https://github.com/dspataru/diabetes-prediction/assets/61765352/81573db1-9ee8-4b87-aad3-b6573da64745)

The best dataset to use for predicting diabetes given the survey data from the CDC was the full dataset: Gen_info_final.

Some important observations:
* Preparing the data is very important for achieving good results
* More features lead to better model performance
* Having a balanced dataset is important to have higher recall and precision for the minority class
* Some features that may not look significant may play a big role in classification
* Future improvements: adding more data for the diabetic patients including sugar level, cholesterol, etc
