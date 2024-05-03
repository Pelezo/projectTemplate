# Santander Customer Satisfaction
The repository holds attempt to apply Machine Learning algorithms to Santander Customer Satisfaction using data from "Banco Santander" Kaggle challenge (https://www.kaggle.com/competitions/santander-customer-satisfaction/submissions#)

# Overview
- Definition of the tasks / challenge: The task defined by the kaggle challenge was to predict if customers are satisfied or unsatisfied based on the feautres affecting the customers satisfaction.
   
- Your approach: The approach in this repository formulates the problem as Regression task using Logistic Regression model and random forest classifier as they are effective models for binary classification tasks. The data was split into training and testing sets. I trained the Logistic Regression model on the training set.
  
- Summary of the performance achieved: Evaluated the model's performance on the testing set using metrics such as accuracy, and roc auc. The Logistic Regession model was able to predict that most of the cutomers were satisfied with an accuracy of 96% and roc auc of 50%. The Random Forest Classifier had similar percentage.  

# Summary of Workdone

# Data
- Data:
    - Type: 
         - Input: CSV file of features, output: satisfied/unsatisfied flag in the last column
    - Size: 215.2 MB, only used half of the data because it is a large dataset. Columns:76020, rows: 371
    - Instances (Train, Test, Validation Split): how many data points? 48652 features for training  for training, 15204 for testing, 12164 for validation. 75818 using the testing 
    dataset. 
#  Preprocessing / Clean up
  - The dataset did not contain any missing values, no duplicates records. The only cleaning performed was to remove the outliers using the Z-score method. 
#  Data Visualization
Show a few visualization of the data and say a few words about what you see.
![image](https://github.com/Pelezo/projectTemplate/assets/143844196/ebbbbe44-9f4e-444c-9637-44f22dde4d79)
![image](https://github.com/Pelezo/projectTemplate/assets/143844196/15a5ec75-972e-44a6-ac76-265ae263f7a2)
![image](https://github.com/Pelezo/projectTemplate/assets/143844196/10411b8a-8db3-4770-b10f-92bb5b1e3e0a)
![image](https://github.com/Pelezo/projectTemplate/assets/143844196/99d9077a-1c4d-4de9-bffa-068422e03d01)


The Histrogram shows distributions of different features by class. Class 1 represents unsatisfied and class 0 represents the satisfied. It feature shows clear separation between the two classes. Also, the classes are unbalanced there is more of class 0 than 1. Also, the histogram are before and after rescalling.

# Problem Formulation
  - Define:
     - Input
        - Train_set.csv: A CSV file containing the training data.
        - Test.csv: A CSV file containing the test data.
     - Output
        - submission.csv: A CSV file containing the predicted target values for 
        the test data.

     - Models
       The models tried were Logistic regression and the Random forest Classifier. I tried 
       both models because they are good for binary classification and also I 
       wanted to see which one would perform better.

# Training
  - Describe the training:
    - **Logistic Regression:**
        - A Logistic Regression model is instantiated.
        - The model is trained using the `fit()` method with the training data (`X_train` and `y_train`).
    - **Random Forest Classifier:**
        - A Random Forest Classifier model is instantiated.
        - The model is trained using the `fit()` method with the training data (`X_train` and `y_train`).
  - Any difficulties? How did you resolve them?
        - There were no issues while training. 
  - Performance Comparison
        - Evaluate the Logistic regression model I used the Roc AUC, Accuracy and Validation Score. For the Random Forest Classifier I used the Accuracy, Precision, F-1 Validation 
        score and test score. In the Logistic regression model the accuracy score was good indicating the model performed well, but ROC AUC was low indicating that the model is not 
        able to learn any meaningful patterns in the data that can be used to predict the class of new examples. Random forest Classifier I got good results except for F-1 and the 
        precision 
# Performance comparison
  - Visualization
  ![image](https://github.com/Pelezo/projectTemplate/assets/143844196/2cb245c6-485d-4a10-90de-02a0a05b8793)

  - Show/compare results in one table.
   ![image](https://github.com/Pelezo/projectTemplate/assets/143844196/7a00a727-4801-4e85-a9ed-eb8adee69fa9)


 # Conclusions
    Machine Learning algorithm is good at making predictions and it can estimate future outcomes based on the data that we have. 
 # Future Work
        - Trying different hyperparameters for the model.
        - Trying different features or feature engineering techniques.
        - Trying different machine learning algorithms.
  - What are some other studies that can be done starting from here.
        - Analyze the results of the model to identify any patterns or trends in 
        the data.
        - This can help to gain a better understanding of the underlying relationships between the features and the target variable.
 # How to reproduce results
     - Set up your environment
     - Prepare your data
     - Train the model
     - Evaluate the model
 # Overview of files in repository
    - DataUnderstandingKagg.ipynb: Loads the data and takes the first look to the dataset. 
    - DataVisualizationKag.ipynb: Creates some Visualization of the data to help understand the data better.
    - KaggleChallenge.ipynb: Prepares the data for the machine learning, creates and evaluates the models. 
 # Software Setup
   - Libraries:
     Pandas 
     Numpy
     Matplotlib
     tabulate
     HTML
     Scikit Learn
     Csv
 # Data
   https://www.kaggle.com/competitions/santander-customer-satisfaction
  
 # Citations
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
 
  
    

