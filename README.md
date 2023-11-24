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

The [BRFSS Overview](https://www.cdc.gov/brfss/annual_data/2022/pdf/Overview_2022-508.pdf) document provides context for what was in the 2022 question set and descriptions of their samples including the target population, geographic populations of interest, and more. They also describe their interviewing procedure for data collection, and how they processed their data. The BRFSS prepare codebooks that contain the variable names for each column in the dataset, which column number corressponds to that variable, what question was asked for that column, and the the values are, along with much more information in a table format.

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

Following columns were selected for the analysis. The values in the columns were analyzed and data cleaning was performed
<br>

| Category         | Renamed-as    | Label/Question                                                                                                                                                                           | Value                                                                                                                               | Null/Refused                               |
| ---------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| <b>\_STATE</b>   | STATE         | <i>-State FIPS Code</i>                                                                                                                                                                  | -Integer [1-78]                                                                                                                     | --                                         |
| <b>DISPCODE</b>  | DISPCODE      | <i>- Final Disposition</i>                                                                                                                                                               | 1100 : Completed Interview<br>1200 : Partial Complete Interview                                                                     | --                                         |
| <b>SEXVAR</b>    | GENDER        | <i>-Sex of Respondent</i>                                                                                                                                                                | 1: MALE<br>2: FEMALE                                                                                                                | --                                         |
| <b>\_INCOMG1</b> | INCOME        | <i>-Income categories</i> (Computed income categories)                                                                                                                                   | Integer [1-7]                                                                                                                       | 9: Don't Know/refused                      |
| <b>HEIGHT3</b>   | HEIGHT        | <i>-About how tall are you without shoes?</i> (Height in Feet and Inches)                                                                                                                | 200 - 711 : ft/inches<br>9061 - 9998 : m/cm                                                                                         | 7777 & 9999 : Don't Know/refused <br>BLANK |
| <b>WTKG3</b>     | WEIGHT        | <i>-Computed Weight in Kilograms</i> (Reported in kilograms)</i>                                                                                                                         | FLOAT [2300 - 29500]                                                                                                                | BLANK                                      |
| <b>\_BMI5CAT</b> | BMI           | <i>-Computed body mass index categories</i> (Four-categories of BMI)</i>                                                                                                                 | 1: Underweight<br>2 : Normal Weight<br>3: Over Weight<br>4: Obese                                                                   | BLANK                                      |
| <b>\_RACE1</b>   | RACE          | <i>-Computed Race-Ethnicity grouping</i> (Race/ethnicity categories)</i>                                                                                                                 | 1: White<br>2: Black<br>3: Indian/ Alaskan Native <br>4: Asian<br>5: Hawaiian/Pacific Islander<br>7: Multiracial<br>8: Hispanic<br> | 9: Don't Know/refused<br>BLANK             |
| <b>\_AGEG5YR</b> | AGE           | <i>Reported age in five-year age categories calculated variable</i>(Fourteen-level age category)                                                                                         | -Integer [1-13]                                                                                                                     | 14: Don't Know/refused                     |
| <b>DIABETE4</b>  | DIABETES      | <i>(Ever told) (you had) diabetes? </i> (If ´Yes´ and respondent is female, ask 'Was this only when you were pregnant?'                                                                  | 1: Yes<br>2: Yes: Only during Pregnancy<br>3: No <br>4: No, Pre-diabeteic/ Border-line                                              | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>PHYSHLTH</b>  | PHYSHLTH      | <i>Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?</i>                 | Num of days [1-30]<br>88 : None                                                                                                     | 77 or 99: Don't Know/refused<br>BLANK      |
| <b>MENTHLTH</b>  | MENTHLTH      | <i> Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?</i> | Num of days [1-30]<br>88 : None                                                                                                     | 77 or 99: Don't Know/refused<br>BLANK      |
| <b>\_TOTINDA</b> | EXERCISE      | <i>Leisure Time Physical Activity Calculated Variable</i>                                                                                                                                | 1: Yes<br>2: No                                                                                                                     | 9: Don't Know/refused                      |
| <b>SLEPTIM1</b>  | SLEEP         | <i>How Much Tim You Sleep</i> On average, how many hours of sleep do you get in a 24-hour period?                                                                                        | Num of hours [1-24]                                                                                                                 | 77 or 99: Don't Know/refused<br>BLANK      |
| <b>PRIMINSR</b>  | HLT_INSURANCE | <i> What is the Doe current primary source of your health insurance?</i>                                                                                                                 | Different Plans [1-10]<br>88: No Coverage                                                                                           | 99: Refused <br>BLANK                      |
| <b>PERSDOC3</b>  | PERSONAL_DOC  | <i> Do you have Personal Health Care Provider?</i>                                                                                                                                       | 1: Yes - Only One<br>2: Yes - More than One <br>3: No                                                                               | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>CHECKUP1</b>  | CHECKUP1      | <i> Length of time since last routine checkup</i>                                                                                                                                        | 1: Within past year<br>2: Within past 2 years <br>3: Within past 5 years<br> 4: 5 or more years ago                                 | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>CVDINFR4</b>  | HRT_ATTACK    | <i>Ever Diagnosed with Heart Attack?</i>                                                                                                                                                 | 1: Yes<br>2: No                                                                                                                     | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>CVDCRHD4</b>  | HRT_DISEASE   | <i>Ever Diagnosed with Angina or Coronary Heart Disease</i>                                                                                                                              | 1: Yes<br>2: No                                                                                                                     | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>CVDSTRK3</b>  | STROKE        | <i>Ever Diagnosed with a Stroke</i>                                                                                                                                                      | 1: Yes<br>2: No                                                                                                                     | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>HAVARTH4</b>  | ARTHRITIS     | <i> (Ever told) (you had) some form of arthritis, rheumatoid arthritis, gout, lupus, or fibromyalgia? </i>                                                                               | 1: Yes<br>2: No                                                                                                                     | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>DIFFWALK</b>  | DIFFWALK      | <i> Do you have serious difficulty walking or climbing stairs?</i>                                                                                                                       | 1: Yes<br>2: No                                                                                                                     | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>\_SMOKER3</b> | \_SMOKER3     | <i>Computed Smoking Status</i> (Four-level smoker status: Everyday smoker, Someday smoker, Former smoker, Non-smoker)                                                                    | 1 : Smokes every day<br>2 : Smokes some days<br>3 : Former Smoker<br>4: Never                                                       | 9: Don't Know/refused                      |
| <b>\_EDUCAG</b>  | EDUCATION     | <i>Computed level of education completed categories</i>                                                                                                                                  | 1 : Did not Graduate High School<br>2 : Graduated High School<br>3 : Attended College<br>4: Graduated from College                  | 9: Don't Know/refused                      |
| <b>CVDSTRK3</b>  | STROKE        | <i>Ever Diagnosed with a Stroke</i>                                                                                                                                                      | 1: Yes<br>2: No                                                                                                                     | 7 or 9: Don't Know/refused<br>BLANK        |
|                  |               | <b>COLUMNS SPECIFIFC TO DIABETIC PATIENTS</b>                                                                                                                                            |                                                                                                                                     |                                            |
| <b>DIABTYPE</b>  | DIABTYPE      | <i> According to your doctor or other health professional, what type of diabetes do you have?</i>                                                                                        | 1: Type 1<br>2: Type 2                                                                                                              | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>PREDIAB2</b>  | PREDIABETIC   | <i> Ever been told by a doctor or other health professional that you have pre-diabetes or borderline diabetes?</i>                                                                       | 1: Yes<br>2: Yes: Only during Pregnancy<br>3: No                                                                                    | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>PDIABTS1</b>  | BLD_SUG_TST   | <i> When was your last blood test for high blood sugar?</i>                                                                                                                              | Integer [1-6] : Past Years<br> 8: Never                                                                                             | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>INSULIN1</b>  | INSULIN_Y/N   | <i>Are you now taking insulin?</i>                                                                                                                                                       | 1: Yes<br>2: No                                                                                                                     | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>CHKHEMO3</b>  | A-one-C_test  | <i>About how many times in the past 12 months has a doctor, nurse, or other health professional checked you for A-one-C?</i>                                                             | Number of times [1-76]<br>88: Never<br>98: Never heard of                                                                           | 77 or 99: Don't Know/refused<br>BLANK      |
| <b>EYEEXAM1</b>  | EYEEXAM1      | <i>When was the last time you had an eye exam in which the pupils were dilated, making you temporarily sensitive to bright light?</i>                                                    | Integer [1-4] : Different Time Range<br>8: Never                                                                                    | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>DIABEYE1</b>  | DIABEYE1      | <i>When was the last time a they took a photo of the back of your eye?</i>                                                                                                               | Integer [1-4] : Different Time Range<br>8: Never                                                                                    | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>DIABEDU1</b>  | DIAB_MNGMT    | <i>When was the last time you took a course or class in how to manage your diabetes yourself?</i>                                                                                        | Integer [1-6] : Past Years<br>8: Never                                                                                              | 7 or 9: Don't Know/refused<br>BLANK        |
| <b>FEETSORE</b>  | FEETSORE      | <i>Have you ever had any sores or irritations on your feet that took more than four weeks to heal?</i>                                                                                   | 1: Yes<br>2: No                                                                                                                     | 7 or 9: Don't Know/refused<br>BLANK        |

## Methodology

### Data Cleaning and Feature Engineering:

**1: Removing all the partially complete inteviews.**
The column `DISPCODE` contains two values : 1200 - For complete interview & 1100 for partially complete interviews. Selected only the rows that contain value `1200`. This reduced the dimension of the data by deleteing 91,861 values.

**2: Editing the values of categorical columns.**
Columns `SEXVAR`, `_BMI5CAT`, `_RACE1`, `_EDUCAG`, `_SMOKER3` contains numerical values, and each value represents a category. These values were changed to categorical values that they represent - Value of 'SEXVAR'- 1: MALE, 2: FEMALE - Value of '\_BMI5CAT'- 1: Underweight, 2 : Normal_Weight, 3: Over_Weight, 4: Obese - Value of '\_RACE1'- 1: "White", 2: "Black", 3: "Indian-Alaskan_Native", 4: "Asian", 5: "Hawaiian", 7: "Multiracial", 8: "Hispanic", 9: np.nan - Value of '\_EDUCAG'- 1: "Grad_HS_N", 2: "Grad_HS_Y", 3: "College_N", 4: "College_Y", 9: np.nan - Value of '\_SMOKER3' - 1: "Smok_daily_Y", 2: "Smok_daily_N", 3: "Prev_Smoker", 4: "Never", 9: np.nan

**3: Data cleaning of Numerical Columns**<br>

- **Calculating the values of HEIGHT Column**

  - For column "HEIGHT3" <br> - 200 - 711: Height (ft/inches)<br>
    --Notes: 0 _ / _ _ = feet / inches-- i.e. the first digit is feet, the second and third digits are inches. So 509 is 5' 9''<br> - 9061 - 9998 Height (meters/centimeters)<br>
    Notes: The initial ´9 ´ indicates this was a metric value. Height in m/cm (9_|\_ \_)<br> - 7777: Don’t know/Not sure<br> - 9999: Refused<br>
  - All the values ranging in between 200 - 711, were first converted the strings, the first letter of the string was taken as feet and converted to float and the last two letters were stored into a variable inches and converted to float. The feet value was multiplied with 12 and added to the inches variable. Another column named as `HEIGHT` was created and the claculated values were stored in that column
  - Values ranging from 9061 to 9998, were also converted to strings, the second letter was taken as meter and last two letters as 'cm', these values were converted to float and 'cm' variable was multiplied by 0.01 and added to m, This gave us height in meters and to convert it into strings the meter value was multiplied by 39.37. this value was also stored in `HEIGHT` column
  - Values 9999 and 7777 were changed to NaN values

- **Substituting the values of 'Don’t know/Not sure' or 'Refused' as NaN values**

  - For some columns the value of 'Don’t know/Not sure' or 'Refused' value is 7 or 9, these values were replaced to NaN values. Column names are
    - "DIABETE4", "PREDIAB2", "DIABTYPE", "\_TOTINDA", "PERSDOC3", "CHECKUP1", "PDIABTS1", "INSULIN1", "EYEEXAM1", "DIABEYE1", "DIABEDU1", "FEETSORE", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "HAVARTH4","DIFFWALK" with 'Don’t know/Not sure' or 'Refused' value as 7 or 9
  - For other columns the value of 'Don’t know/Not sure' or 'Refused' value as 77 or 99, these were also replaced with NaN values. Column names are:
    - "CHKHEMO3", "PHYSHLTH", "MENTHLTH", "SLEPTIM1", "PRIMINSR"
  - For column "\_AGEG5YR", value 14 represents 'Don’t know/Not sure' value, it was converted to NaN value

- **Substituting the values of No/Never with 0**
  - All the values for No/Never were converted to 0.
    - For some columns ("\_TOTINDA", "INSULIN1", "FEETSORE", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "HAVARTH4","DIFFWALK") the value was 2
    - For some ("PERSDOC3", "DIABETE4", "PREDIAB2") the value was 3
    - For some ("CHECKUP1", "PDIABTS1", "EYEEXAM1","DIABEYE1", "DIABEDU1") the value was 8
    - For others ("CHKHEMO3", "PHYSHLTH", "MENTHLTH", "SLEPTIM1", "PRIMINSR") the value was 88

**4: Saving the modified datframe as .csv, keeping the index column.** This index column is renamed as `ID` column. This entire data has to be divided into two dataframes:

**These two dataframes can later be joined on the "ID Column"** -- This was done for the stacked machine learning model

**5: Splitting the data into two dataframes:**

- `Index` column was renamed as `ID`
- Columns were re-named for a clear understanding of the data
- Null values were counted for each column -- a few columns have question specific to the patients who have diabetes therefore they all have same number of missing values (questions that were not asked from the non-diabetic people).

![Alt text](image.png)

- Therefore, a two separate dataframes are created:

  - Dataframe 1: Containing the general information <br>
    Will be used to categorize if the person has diabetes or not. Columns specific to Diabetes are dropped new dataframe "gen_info_df"
  - Dataframe 2: Specific to Diabetic patients <br>
    Will be used to categorize the type of diabetes (type 1 or type 2)

- **From general information dataframe** all the null values were deleted. This resulted in a dataframe of dimension - 246050 rows X 24 columns. This clean data was saved to a .CSV file and to the AWS server
- **From Diabetes Specific dataframe**
  - All the rows representing no diabetes were dropped
  - There were still null values in the Type of Diabetes column, all the null values from this column were also dropped. This resulted in a dataframe of dimension 9975 rows X 19 columns
  - All the values in these two columns: `PREDIABETIC` and `BLD_SUG_TST`were Null, so these columns were dropped
  - "NaN' values were checked for each column: This dataframe already had a very small subset, therefore, other null values were replaced either by mean or randomly
  - **Feature Engineering for Diabetes Dataframe**
    - Since `AGE` and `BMI` categories cannot be 0, the NaN values in 'AGE' column were replaced with the mean value of the age
    - `BMI` being a categorical column, the NaN values were replaced using `random` function that randomly selected category based on the distribution of existing values
    - For all other columns the NaN values were replaced by 0, which signifies "NO/NEVER"
    - This resulted in a dataframe of dimension - 9975 rows × 17 columns. This clean data was saved to a .CSV file and to the AWS server
    

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
