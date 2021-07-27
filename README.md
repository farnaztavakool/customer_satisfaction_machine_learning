# customer_satisfaction_machine_learning
## How to run 
1)you can install the dependencies with pip3 install -r requirements.txt
2) you can run the python code with python3

## current outuput
<img width="568" alt="image" src="https://user-images.githubusercontent.com/60339514/126883739-089cd32e-9daf-4b0f-a00e-6e1fcb2a5e41.png">


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
