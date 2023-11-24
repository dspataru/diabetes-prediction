# diabetes-prediction


# Feature Analysis

1. Logistic Regression and  Random Rainforest

As per the picture provided, it is evident that the likelihood of having diabetes or not having diabetes is directly related to the age of the person. Approximately 25% of the dataset is covered by age, followed by income and general health. This insight emphasizes the significance of age in understanding and predicting diabetes in the given dataset.
￼

 ![Features Importances](https://github.com/dspataru/diabetes-prediction/assets/136105558/2cacf8d8-3b2a-4e54-a204-6956e858614c)

 
![Features Importances](https://github.com/dspataru/diabetes-prediction/assets/136105558/581baa07-e32f-4370-aad8-d332d7686632)

# Model Description:

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

# Analysis of Logistic Regression and Random Rainforest

The dynamic duo of logistic regression and random forests. Imagine logistic regression as a savvy detective making decisions based on clues—it predicts outcomes, like whether someone has a diabetes or not. Now, pair that with random forests—it's like having a team of detectives, each with a unique perspective, working together to crack the case. The combination is powerful! Logistic regression sets the stage, and random forests bring diversity to the decision-making process, enhancing accuracy. Together, they're our tech detectives, for making sense of complex data and delivering reliable predictions.

In this example, we first load the diabetes dataset using pandas library. Then we split the data into training and testing sets using train_test_split function. Next, we create a logistic regression model using LogisticRegression class and train the model on the training data using fit method. Finally, we make predictions on the testing data using predict method and evaluate the model's accuracy using accuracy_score function.

So, I went ahead and tested them in two data set:- one where the data set is divided equally and another one being under and over sampling! 

Classification Reports and ROC Charts are:- 

1. Equal Sampling: 

   1.1 Classification Report:


   1.2 ROC Chart:


 2. Over Sampling and Under Sampling:

   2.1 Classification Report:

 
    
   2.2 ROC Chart:

