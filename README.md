# Customer Satisfaction Model for Santander Bank

Our project was chosen from a Kaggle competition: https://www.kaggle.com/c/santander-customer-satisfaction

Our aim for this project is to build a set of models to classify whether a customer is satisfied or dissatisfied with Santander Bank. 

## Files provided by the Challenge

We were provided with a train.csv and test.csv data file. 

All the features in the dataset are fully anonymized, leaving the TARGET variable with values of 0 and 1 where 1 indicates an unhappy customer, and 0 indicates a satisfied customer. 

# Dependencies
1) Python3 and pip3 are required to run the code.
2) Install the dependencies by typing 'pip3 install -r requirements.txt'.

## Jupyter notebooks
because we are using deep learning we included  jupyter notebooks that have cached results for Neural Network, KNN, Logistic regression, DecisionTree and the main jupyter notebook that essemble different models.

## What are the different files and what they do
1) There are seperate files and cached jupyter notebooks for each model which are called: 
   * neural_network.py
   * KNN.py
   * decisionTree.py
   * logisticRegression.py 
   * neural_network.ipynb 
   * decisionTree.ipynb
   * logisticRegression.ipynb
   * KNN.ipynb 
2) main.py: This is the main file in the codebase becaus it trains all of the models and fit them based on the hyperparameters that we found and assemble them to produce the final prediction.
3) run.py: Because we are having multiple files, we created a run.py file to help run different files in codebase
4) data_preprocess.py: This file has all of the data cleaning, preprocessing and feature_selection methods and will save the preprocessed data into seperate files called "X_train.csv", "X_test.csv", "Y_train.csv".
5) EDA.py: This file will produce multiple graphs that have been used in the report to analyse the data

# How to run after installing the dependencies (prefered method)

1) First change to src folder and
> cd src 

2) Then run data_preprocess.py
> python3 data_preprocess.py


3) Use the following command to run the main file
> python3 run.py main

4) To run the models seperately via commands:
* For KNN model:
*   > python3 run.py knn find_best_k_value
*   > python3 run.py knn train_KNN_model
*   > python3 run.py knn K_value_tuning <X_validation> <Y_validation>
* For DecisionTree model:
*   > python3 run.py tree train_decisionTree_model
*   > python3 run.py tree find_best_depth
*   > python3 run.py tree test_decisionTree
*   > python3 run.py tree depth_tuning <X_validation> <Y_validation>
* For LogisticRegression model:
*   > python3 run.py log train_lgr_model
*   > python3 run.py log find_best_solver
*   > python3 run.py log find_best_C
*   > python3 run.py log c_value_tuning <X_validation> <Y_validation>
* For NeuralNetwork model:
*   > python3 run.py net
* For Generating graphs of 'Exploratory-Data-Analysis for Santander Bank'
*   > python3 run.py eda
* Note: 
* > python3 run.py     (Displays the usage options above)
* > python3 run.py knn    (Displays the usage options for knn only)

## How to run after installing the dependencies (original method)

1) first change to src folder 
> cd src
2) run data_preprocess.py
> python3 data_preprocess.py  
3) run main.py
> python3 main.py 



