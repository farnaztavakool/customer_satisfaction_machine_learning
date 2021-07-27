# customer_satisfaction_machine_learning
## How to run 
1)you can install the dependencies with pip3 install -r requirements.txt
2) you can run the python code with python3

## current outuput
<img width="568" alt="image" src="https://user-images.githubusercontent.com/60339514/126883739-089cd32e-9daf-4b0f-a00e-6e1fcb2a5e41.png">


## data analysis:

## data preprocess:
    I separate the data preprocess so that we don't need to preprocess data every time we train our model. However, the output of the csv files are too large
    to push to github repo. So just run data_preprocess.py and it will export the training and test set. I also added these csv files into .gitignore so that 
    we won't commit it accidentlly.
        X_train.csv: the input of training set
        Y_train.csv: the output of training set
        X_test.csv: the input of testing set

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


## KNN model:
    Score for submission: 0.65 n_neighbor = 5
    The training for KNN model is really slow, have no idead why.

    Since the data is highly unbalance (around 96% of the target is 0 and 4% of the target is 1). If we use a large k value for our KNN model, every sample will be predicted as 0 since 
    we have a really large propotion of 0s. And also this will not decrease our accuracy since even though we output every target as 0, we will still get a high accuracy around 96%.



## decision Tree model:
    Score for submission: 0.71 (depth = 20)

## A combination of KNN and decision Tree model:
    Score for submission: 0.749