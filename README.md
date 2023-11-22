# diabetes-prediction

![diabetes](https://github.com/dspataru/diabetes-prediction/assets/61765352/5b2eb604-ef51-4f85-b050-136fedfd6756)


## Table of Contents
* [Introduction](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#introduction)
* [Background](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#background)
* [Data Source](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#data-source)
* [Methodology](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#methodology)
* [Analysis and Results](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#analysis-and-results)
* [Conclusion](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#conclusion)
* [Future Work](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#future-work)
* [References](https://github.com/dspataru/diabetes-prediction/blob/main/README.md#references)

## Introduction

In recent years, the intersection of healthcare and machine learning has emerged as a transformative field, offering innovative solutions to longstanding challenges. Among the myriad applications, predicting and managing chronic diseases, such as diabetes, has become a focal point. Diabetes is a global health concern affecting millions of people around the world, and necessitates accurate and timely predictive tools for early intervention and personalized treatment.

The integration of machine learning models in healthcare has played a pivotal role in advancing diagnostic and predictive capabilities. Among these models, classification algorithms can be a useful tool for predicting the onset of diabetes. This report delves into the significance of employing machine learning models, particularly classification models, in predicting diabetes. By exploring the differences between prominent classification models such as K-nearest neighbors, neural networks, and random forests, this report aims to find a model that is most accurate at predicting diabetes.

Understanding the intricacies of these models is crucial, as they leverage distinct methodologies to analyze patient data and make predictions. K-nearest neighbors relies on proximity-based relationships, neural networks emulate the complex structure of the human brain, and random forests employ ensemble learning to enhance predictive accuracy. Examining these models in tandem allows for a nuanced appreciation of their individual contributions to diabetes prediction.

An essential facet of utilizing classification models in healthcare is the assessment of their performance. This report will explore various metrics and techniques employed to evaluate the efficacy of these models, ensuring that the chosen algorithms meet the stringent requirements of reliability and accuracy demanded by the healthcare domain. Furthermore, optimization strategies will be discussed, elucidating ways to fine-tune these models for enhanced performance in real-world scenarios.

In summary, this exploration aims to empower healthcare professionals and data scientists alike in their pursuit of accurate and efficient diabetes prediction models.

#### Libraries used
psycopg2, sqlalchemy, pandas, numpy, sklearn, tensorflow, matplotlib, seaborn

## Background

## Data Source

Following columns were selected for the analysis. The values in the columns were analyzed and data cleaning was performed.
<br>

|Category|Renamed-as|Label/Question|Value|Null/Refused| 
|-|-|-|-|-|
|<b>_STATE</b>|STATE|<i>-State FIPS Code</i>|-Integer [1-78]|--|
|<b>DISPCODE</b>|DISPCODE|<i>- Final Disposition</i>|1100 : Completed Interview<br>1200 : Partial Complete Interview|--|
|<b>SEXVAR</b>|GENDER|<i>-Sex of Respondent</i>|1: MALE<br>2: FEMALE|--|
|<b>_INCOMG1</b>|INCOME|<i>-Income categories</i> (Computed income categories)|Integer [1-7]|9: Don't Know/refused|
|<b>HEIGHT3</b>|HEIGHT|<i>-About how tall are you without shoes?</i> (Height in Feet and Inches)|200 - 711 : ft/inches<br>9061 - 9998 : m/cm |7777 & 9999 : Don't Know/refused <br>BLANK|
|<b>WTKG3</b>|WEIGHT|<i>-Computed Weight in Kilograms</i> (Reported in kilograms)</i>|FLOAT [2300 - 29500]|BLANK|
|<b>_BMI5CAT</b>|BMI|<i>-Computed body mass index categories</i> (Four-categories of BMI)</i>|1: Underweight<br>2 : Normal Weight<br>3: Over Weight<br>4: Obese|BLANK|
|<b>_RACE1</b>|RACE|<i>-Computed Race-Ethnicity grouping</i> (Race/ethnicity categories)</i>|1: White<br>2: Black<br>3: Indian/ Alaskan Native <br>4: Asian<br>5:  Hawaiian/Pacific Islander<br>7: Multiracial<br>8: Hispanic<br>|9: Don't Know/refused<br>BLANK|
|<b>_AGEG5YR</b>|AGE|<i>Reported age in five-year age categories calculated variable</i>(Fourteen-level age category)|-Integer [1-13]|14: Don't Know/refused
|<b>DIABETE4</b>|DIABETES|<i>(Ever told) (you had) diabetes? </i> (If ´Yes´ and respondent is female, ask 'Was this only when you were pregnant?'|1: Yes<br>2: Yes: Only during Pregnancy<br>3: No <br>4: No, Pre-diabeteic/ Border-line|7 or  9: Don't Know/refused<br>BLANK|
|<b>PHYSHLTH</b>|PHYSHLTH|<i>Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?</i> |Num of days [1-30]<br>88 : None|77 or  99: Don't Know/refused<br>BLANK|
|<b>MENTHLTH</b>|MENTHLTH|<i> Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?</i>|Num of days [1-30]<br>88 : None|77 or  99: Don't Know/refused<br>BLANK|
|<b>_TOTINDA</b>|EXERCISE|<i>Leisure Time Physical Activity Calculated Variable</i>|1: Yes<br>2: No| 9: Don't Know/refused|
|<b>SLEPTIM1</b>|SLEEP|<i>How Much Tim You Sleep</i> On average, how many hours of sleep do you get in a 24-hour period?|Num of hours [1-24]|77 or  99: Don't Know/refused<br>BLANK|
|<b>PRIMINSR</b>|HLT_INSURANCE|<i> What is the Doe current primary source of your health insurance?</i>|Different Plans [1-10]<br>88: No Coverage |99: Refused <br>BLANK|
|<b>PERSDOC3</b>|PERSONAL_DOC|<i> Do you have Personal Health Care Provider?</i>|1: Yes - Only One<br>2: Yes -  More than One <br>3: No |7 or  9: Don't Know/refused<br>BLANK|
|<b>CHECKUP1</b>|CHECKUP1|<i> Length of time since last routine checkup</i>|1: Within past year<br>2: Within past 2 years <br>3: Within past 5 years<br> 4: 5 or more years ago  |7 or  9: Don't Know/refused<br>BLANK|
|<b>CVDINFR4</b>|HRT_ATTACK|<i>Ever Diagnosed with Heart Attack?</i>|1: Yes<br>2: No|7 or  9: Don't Know/refused<br>BLANK|
|<b>CVDCRHD4</b>|HRT_DISEASE|<i>Ever Diagnosed with Angina or Coronary Heart Disease</i>|1: Yes<br>2: No|7 or  9: Don't Know/refused<br>BLANK|
|<b>CVDSTRK3</b>|STROKE|<i>Ever Diagnosed with a Stroke</i>|1: Yes<br>2: No|7 or  9: Don't Know/refused<br>BLANK|
|<b>HAVARTH4</b>|ARTHRITIS|<i> (Ever told) (you had) some form of arthritis, rheumatoid arthritis, gout, lupus, or fibromyalgia? </i>|1: Yes<br>2: No|7 or  9: Don't Know/refused<br>BLANK|
|<b>DIFFWALK</b>|DIFFWALK|<i> Do you have serious difficulty walking or climbing stairs?</i>|1: Yes<br>2: No|7 or  9: Don't Know/refused<br>BLANK|
|<b>_SMOKER3</b>|_SMOKER3|<i>Computed Smoking Status</i> (Four-level smoker status:  Everyday smoker, Someday smoker, Former smoker, Non-smoker) |1 : Smokes every day<br>2 : Smokes some days<br>3 : Former Smoker<br>4: Never|9: Don't Know/refused|
|<b>_EDUCAG</b>|EDUCATION|<i>Computed level of education completed categories</i> |1 : Did not Graduate High School<br>2 : Graduated High School<br>3 : Attended College<br>4: Graduated from College |9: Don't Know/refused|
|<b>CVDSTRK3</b>|STROKE|<i>Ever Diagnosed with a Stroke</i>|1: Yes<br>2: No|7 or  9: Don't Know/refused<br>BLANK|
|||<b>COLUMNS SPECIFIFC TO DIABETIC PATIENTS</b>|||
|<b>DIABTYPE</b>|DIABTYPE|<i> According to your doctor or other health professional, what type of diabetes do you have?</i>|1: Type 1<br>2: Type 2 |7 or  9: Don't Know/refused<br>BLANK|
|<b>PREDIAB2</b>|PREDIABETIC|<i> Ever been told by a doctor or other health professional that you have pre-diabetes or borderline diabetes?</i>|1: Yes<br>2: Yes: Only during Pregnancy<br>3: No |7 or  9: Don't Know/refused<br>BLANK|
|<b>PDIABTS1</b>|BLD_SUG_TST|<i> When was your last blood test for high blood sugar?</i>|Integer [1-6] : Past Years<br> 8: Never|7 or  9: Don't Know/refused<br>BLANK|
|<b>INSULIN1</b>|INSULIN_Y/N|<i>Are you now taking insulin?</i>|1: Yes<br>2: No|7 or  9: Don't Know/refused<br>BLANK|
|<b>CHKHEMO3</b>|A-one-C_test|<i>About how many times in the past 12 months has a doctor, nurse, or other health professional checked you for A-one-C?</i>|Number of times [1-76]<br>88: Never<br>98: Never heard of|77 or  99: Don't Know/refused<br>BLANK|
|<b>EYEEXAM1</b>|EYEEXAM1|<i>When was the last time you had an eye exam in which the pupils were dilated, making you temporarily sensitive to bright light?</i>|Integer [1-4] : Different Time Range<br>8: Never|7 or  9: Don't Know/refused<br>BLANK|
|<b>DIABEYE1</b>|DIABEYE1|<i>When was the last time a they took a photo of the back of your eye?</i>|Integer [1-4] : Different Time Range<br>8: Never|7 or  9: Don't Know/refused<br>BLANK|
|<b>DIABEDU1</b>|DIAB_MNGMT|<i>When was the last time you took a course or class in how to manage your diabetes yourself?</i>|Integer [1-6] : Past Years<br>8: Never|7 or  9: Don't Know/refused<br>BLANK|
|<b>FEETSORE</b>|FEETSORE|<i>Have you ever had any sores or irritations on your feet that took more than four weeks to heal?</i>|1: Yes<br>2: No|7 or  9: Don't Know/refused<br>BLANK|


## Methodology


### Data Cleaning and Feature Engineering:
**1: Removing all the partially complete inteviews.**
The column `DISPCODE` contains two values  :  1200 - For complete interview & 1100 for partially complete interviews. Selected only the rows that contain value `1200`. This reduced the dimension of the data by deleteing 91,861 values.

**2: Editing the values of categorical columns.**
Columns `SEXVAR`, `_BMI5CAT`, `_RACE1`, `_EDUCAG`, `_SMOKER3` contains numerical values, and each value represents a category. These values were changed to categorical values that they represent
    - Value of 'SEXVAR'- 1: MALE, 2: FEMALE
    - Value of '_BMI5CAT'- 1: Underweight, 2 : Normal_Weight, 3: Over_Weight, 4: Obese
    - Value of '_RACE1'- 1: "White", 2: "Black", 3: "Indian-Alaskan_Native", 4: "Asian", 5: "Hawaiian", 7: "Multiracial", 8: "Hispanic", 9:  np.nan
    - Value of '_EDUCAG'- 1: "Grad_HS_N", 2: "Grad_HS_Y", 3: "College_N", 4: "College_Y", 9:  np.nan
    - Value of '_SMOKER3' - 1: "Smok_daily_Y", 2: "Smok_daily_N", 3: "Prev_Smoker", 4: "Never", 9:  np.nan

**3: Data cleaning of Numerical Columns**<br>
  - **Calculating the values of HEIGHT Column**
      - For column "HEIGHT3" <br>
             - 200 - 711: Height (ft/inches)<br>
              --Notes: 0 _ / _ _ = feet / inches-- i.e. the first digit is feet, the second and third digits are inches. So 509 is 5' 9''<br>
              - 9061 - 9998	Height (meters/centimeters)<br>
              Notes: The initial ´9 ´ indicates this was a metric value. Height in m/cm (9_|_ _)<br>
              - 7777:	Don’t know/Not sure<br>	
              - 9999: Refused<br>
    - All the values ranging in between 200 - 711, were first converted the strings, the first letter of the string was taken as feet and converted to float and the last two letters were stored into a variable inches and converted to float. The feet value was multiplied with 12 and added to the inches variable. Another column named as `HEIGHT` was created and the claculated values were stored in that column
    - Values ranging from 9061 to 9998, were also converted to strings, the second letter was taken as meter and last two letters as 'cm', these values were converted to float and 'cm' variable was multiplied by 0.01 and added to m, This gave us height in meters and to convert it into strings the meter value was multiplied by 39.37. this value was also stored in `HEIGHT` column
    - Values 9999 and 7777 were changed to NaN values

- **Substituting the values of 'Don’t know/Not sure' or 'Refused' as NaN values**
    - For some columns the value of 'Don’t know/Not sure' or 'Refused' value is 7 or 9, these values were replaced to NaN values. Column names are
        - "DIABETE4", "PREDIAB2", "DIABTYPE", "_TOTINDA", "PERSDOC3", "CHECKUP1", "PDIABTS1", "INSULIN1", "EYEEXAM1", "DIABEYE1", "DIABEDU1", "FEETSORE", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "HAVARTH4","DIFFWALK" with 'Don’t know/Not sure' or 'Refused' value as 7 or 9
    - For other columns the value of 'Don’t know/Not sure' or 'Refused' value as 77 or 99, these were also replaced with NaN values. Column names are:
      - "CHKHEMO3", "PHYSHLTH", "MENTHLTH", "SLEPTIM1", "PRIMINSR"
    - For column "_AGEG5YR", value 14 represents 'Don’t know/Not sure' value, it was converted to NaN value

- **Substituting the values of No/Never with 0**
    - All the values for No/Never were converted to 0.
        - For some columns ("_TOTINDA", "INSULIN1", "FEETSORE", "CVDINFR4", "CVDCRHD4", "CVDSTRK3", "HAVARTH4","DIFFWALK") the value was 2
        - For some ("PERSDOC3", "DIABETE4", "PREDIAB2") the value was 3
        - For some ("CHECKUP1", "PDIABTS1", "EYEEXAM1","DIABEYE1", "DIABEDU1") the value was 8
        - For others ("CHKHEMO3", "PHYSHLTH", "MENTHLTH", "SLEPTIM1", "PRIMINSR") the value was 88
  

    



## Analysis and Results

## Conclusion

## Future Work

## References
