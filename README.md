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

### Analysis with Dataset 3


## Conclusion

Below is a summary of the results of the best model performance given the different datasets.
![summary-of-results](https://github.com/dspataru/diabetes-prediction/assets/61765352/81573db1-9ee8-4b87-aad3-b6573da64745)

The best dataset to use for predicting diabetes given the survey data from the CDC was the full dataset containing 34 features including 

Some important observations:
* Preparing the data is very important for achieving good results
* More features lead to better model performance
* Having a balanced dataset is important to have higher recall and precision for the minority class
* Some features that may not look significant may play a big role in classification
* Future improvements: adding more data for the diabetic patients including sugar level, cholesterol, etc
