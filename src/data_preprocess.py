from sklearn.tree import DecisionTreeClassifier
import customer_satisfaction as cs
import pandas as pd
from sklearn.utils import resample
import math
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import plot_confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def preprocessData(df_train, df_test):

    dup = cs.dropDuplicatedRowAndColumn(df_train, df_test)
    df_train = dup[0]
    df_test = dup[1]

    y_train = df_train['TARGET'].copy()
    df_train_x = df_train.drop(columns="TARGET")

    quasi_res = cs.quasiConstantRemoval(df_train_x, 0.01, df_test)
    df_train_x = quasi_res[0]
    df_test = quasi_res[1]

    correlated  = cs.dropCorrelatedFeatures(df_train_x,df_test)
    df_train_x = correlated[0]
    df_test = correlated[1]

    # use ANOVA to select k best features, the number of features is a hyperparameter which we need to test to find a best one
    num_features_ANOVA = 100
    df_train_x, df_test = cs.featureSelectionANOVA(df_train_x, y_train, df_test, num_features_ANOVA)

    # use decisionTreeClassifier for the estimator of RFE to fun the feature selection
    estimator = DecisionTreeClassifier()
    num_features_RFE = 70
    df_train_x, df_test = cs.featureSelectionRFE(df_train_x, y_train, df_test, num_features_RFE, estimator)


    df_train_x = cs.normalise(df_train_x)
    df_test = cs.normalise(df_test)
    
    df_train_x.to_csv('X_train.csv', index= False)
    y_train.to_csv('Y_train.csv', index= False)
    df_test.to_csv('X_test.csv', index= False)

    return [df_train_x, y_train, df_test]


def oversampling_dataset(data):
    count_majority, count_minority = data['TARGET'].value_counts()
    
    data_majority=data[data.TARGET==0] 
    data_minority=data[data.TARGET==1]  

    data_minority_over = data_minority.sample(count_majority, replace=True)
    data_oversampled=pd.concat([data_minority_over,data_majority])

    print(data_oversampled['TARGET'].value_counts())
    return data_oversampled

def undersampling_dataset(data):
    count_majority, count_minority = data['TARGET'].value_counts()
    
    data_majority=data[data.TARGET==0]
    data_minority=data[data.TARGET==1]

    data_majority_under = data_majority.sample(count_minority)
    data_underampled=pd.concat([data_majority_under,data_minority])

    print(data_underampled['TARGET'].value_counts())
    return data_underampled

'''
1) getting half of unsatisfied
2) getting half of satisfied 
3) concatenating them to have a representative sample
'''
def consistent_sampling(data):
    
    count_majority, count_minority = data['TARGET'].value_counts()
    data_majority=data[data.TARGET==0] 
    data_minority=data[data.TARGET==1] 

 
    x_train_majority, x_test_majority, y_train_majority, y_test_majority = train_test_split(data_majority.drop(['TARGET'],axis=1), data_majority['TARGET'],train_size=math.floor(count_majority/2))
    x_train_minority, x_test_minority, y_train_minority, y_test_minority = train_test_split(data_minority.drop(['TARGET'],axis=1), data_minority['TARGET'],train_size=math.floor(count_minority/2))
    

    size = math.floor(count_majority/2) + math.floor(count_minority/2)
    x_test = pd.concat([x_test_majority, x_test_minority], axis=0).reset_index(drop=True)
    x_train = pd.concat([x_train_majority, x_train_minority], axis=0)
    y_train = pd.concat([y_train_majority, y_train_minority], axis=0).to_numpy()
    y_test = pd.concat([y_test_majority, y_test_minority], axis=0).to_numpy()
   
  
    return x_train, x_test, y_train, y_test


df_train = cs.loadData("train.csv")
df_test = cs.loadData("test.csv")

# dropping "ID" from both training and test data
ID_train = df_train["ID"].copy()
ID_test = df_test["ID"].copy()
df_train = df_train.drop(columns = "ID")
df_test = df_test.drop(columns = "ID")


