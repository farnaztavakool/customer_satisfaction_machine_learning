# customer_satisfaction_machine_learning
# Dependencies
1) Python3 and pip3 are required to run the code.
2) Install the dependencies by typing 'pip3 install -r requirements.txt'.

## Jupyter notebooks
because we are using deep learning we included  jupyter notebooks that have cached results for Neural Network, KNN, Logistic regression, DecisionTree and the main jupyter notebook that essemble different models.

## What are the different files and what they do
1) There are seperate files for each model which are called: neural_network.py, KNN.py, decisionTree.py, logisticRegression.py and each one of these files have a jupyter notebook.
2) There is a file called main.py. This file trains all of the models and fit them based on the hyperparameters that we found running the model files.
3) Finally, run.py is the file ...
4) 
## How to run 
After installing the dependencies:
1) run data_preprocess.py file by running: python3 data_preprocess.py
2) run main.py
3) To run each of the files seperately you can run:
4) 
5) If there is keras import error, change the import 'keras' with 'tensorflow.keras' in every .py file.
6) You can type 'python3 run.py <file> <function> <parameter_1> <parameter_2>' in terminal to run code seperately.
7) Type 'python3 run.py' will show the Usage
8) In general, type 'python3 run.py main' to run everything
9) Plots are stored under 'src/images' if after running with run.py




## data analysis:

## data preprocess:
    I separate the data preprocess so that we don't need to preprocess data every time we train our model. However, the output of the csv files are too large
    to push to github repo. So just run data_preprocess.py and it will export the training and test set. I also added these csv files into .gitignore so that 
    we won't commit it accidentlly.
        X_train.csv: the input of training set
        Y_train.csv: the output of training set
        X_test.csv: the input of testing set

    Unsampling data:
        our data is highly imbalanced, so we might need to oversample our data.

## Feature selection:
    Filter based method:
        1. pearson correlation: (this is already done by Farnaz in dropCorrelatedFeatures() function)
            this method picks the features which is strongly related to the target value. if a feature has a weak relationship with the target value, then
            it might be an irrelavant feature and so we drop it.
            However, pearson correlation is only sensitive to linear relationship, it the feature has non-linear relationship with the target, it might have
            low pearson correlation so might be classified as irrelavant feature.
        2.Chi-square (done by John):
            chi-square is similar to pearson correlation, but this is for categorical data so it should not be suitable for our dataset.
        3.Variance (also already done by Farnaz):

    Wrapper-based method:
        1. Recursive feature elimination (RFE):
            The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached. --------------------- from sklearn documentation

            so the important part for is to choose the correct estimator.
        2. Analysis of Variance (ANOVA):
            this is a supervised feature selection method which uses f-statistic of each features to determine which feature is important.
            We need to decide how many features to keep in order to get a good prediction.





