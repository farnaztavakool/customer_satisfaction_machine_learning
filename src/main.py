import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, chi2, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, plot_confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from neural_network import get_CV_prediction, build_model
import math
import random
import KNN
import decisionTree
import logisticRegression

def loadData(path):
    return pd.DataFrame(pd.read_csv(path))

def dropDuplicatedRowAndColumn(train, test):   
    train = train.drop_duplicates()
    drop = train.columns.duplicated()
    return [train.loc[:,~drop], test.loc[:,~drop[:len(drop)-1]]]

def quasiConstantRemoval(train, threshold, test):
    constant_filter = VarianceThreshold(threshold= threshold)
    constant_filter.fit(train)
    return [pd.DataFrame(constant_filter.transform(train)),pd.DataFrame(constant_filter.transform(test))]

def dropCorrelatedFeatures(train,test):
    drop_correlated = DropCorrelatedFeatures(
    variables=None, method='pearson', threshold=0.9)
    drop_correlated.fit(train)
    return [pd.DataFrame(drop_correlated.transform(train)),pd.DataFrame(drop_correlated.transform(test))]

def featureSelectionANOVA(train_X, train_Y, test, numOfFeatures):
    fvalue_best = SelectKBest(f_classif, k=numOfFeatures)
    fvalue_best.fit(train_X, train_Y)
    return [pd.DataFrame(fvalue_best.transform(train_X)),pd.DataFrame(fvalue_best.transform(test))]
    
def featureSelectionRFE(train_X, train_Y, test, numOfFeatures, estimator):
    rfe = RFE(estimator = estimator, n_features_to_select=numOfFeatures)
    rfe.fit(train_X, train_Y)
    return [pd.DataFrame(rfe.transform(train_X)),pd.DataFrame(rfe.transform(test))]

def featureSelectionCHI2(train_X, train_Y, test, numOfFeatures):
    new_best = SelectKBest(chi2, k=numOfFeatures)
    new_best.fit(train_X, train_Y)
    return [pd.DataFrame(new_best.transform(train_X)),pd.DataFrame(new_best.transform(test))]

def dropColumnWithName(data, name):
    column_to_drop = [c for c in data if c.startswith(name)]
    data = data.drop(columns = column_to_drop )

def normalise(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data))

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
    
    x_train.insert(x_train.shape[1], 'TARGET', y_train)
    x_test.insert(x_test.shape[1], 'TARGET', y_test)
    
    x_train.sample(frac=1)
    x_test.sample(frac=1)
    
    
  
    return x_train.drop(columns="TARGET"), x_test.drop(columns='TARGET'), x_train['TARGET'], x_test['TARGET']

def evaluate_model(model, X_test, Y_test, model_name):
    if (model_name == 'Neural Network'):
        prediction = model.predict_proba(X_test).ravel()
    else :
        prediction = model.predict_proba(X_test)[:,1]
    false_positive_rate, true_positive_rate, threshold1 = roc_curve(Y_test, prediction)
    print('roc_auc_score for ', model_name,': ', roc_auc_score(Y_test, prediction))

    plt.title('ROC - ' + model_name)
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    if(model_name != 'Neural Network'):
        plot_confusion_matrix(model, X_test, Y_test)
        plt.show()

## --------------------------------------------------------------load data-------------------------------------------
df_train = loadData("train.csv")
df_test = loadData("test.csv")

# dropping "ID" from both training and test data
ID_train = df_train["ID"].copy()
ID_test = df_test["ID"].copy()
df_train = df_train.drop(columns = "ID")
df_test = df_test.drop(columns = "ID")

dup = dropDuplicatedRowAndColumn(df_train, df_test)
df_train = dup[0]
df_test = dup[1]

df_train_y = df_train['TARGET'].copy()
df_train_x = df_train.drop(columns="TARGET")



## --------------------------------------------------------------data preprocess---------------------------------
quasi_res = quasiConstantRemoval(df_train_x, 0.01, df_test)
df_train_x = quasi_res[0]
df_test = quasi_res[1]

correlated  = dropCorrelatedFeatures(df_train_x,df_test)
df_train_x = correlated[0]
df_test = correlated[1]

## ---------------------------------------------------------------feature selection------------------------------
# use ANOVA to select k best features
num_features_ANOVA = 100
df_train_x, df_test = featureSelectionANOVA(df_train_x, df_train_y, df_test, num_features_ANOVA)

# use decisionTreeClassifier for the estimator of RFE to fun the feature selection
estimator = DecisionTreeClassifier()
num_features_RFE = 70
df_train_x, df_test = featureSelectionRFE(df_train_x, df_train_y, df_test, num_features_RFE, estimator)

df_train_x = normalise(df_train_x)
df_train_y = np.ravel(df_train_y)
df_test = normalise(df_test)

## ---------------------------------------------------------------- split data -----------------------------------

# split data into train set, validation set and test set as 2:1:1
'''
df_train_x.insert(df_train_x.shape[1], 'TARGET', df_train_y)
X_train, X_test, Y_train, Y_test = consistent_sampling(df_train_x)

X_test.insert(X_test.shape[1], 'TARGET', Y_test)
X_validation, X_test, Y_validation, Y_test = consistent_sampling(X_test)

print(Y_validation)
'''
X_train, X_test, Y_train, Y_test = train_test_split(df_train_x, df_train_y, test_size = 0.5)

X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.5)

## ---------------------------------------------------------------- hypeparameter tuning -------------------------
'''
KNN.K_value_tuning(X_val, Y_val)
decisionTree.depth_tuning(X_val, Y_val)
logisticRegression.c_value_tuning(X_val, Y_val)
'''
## --------------------------------------------------------------- Train models -----------------------------------

# KNN model
knn = KNeighborsClassifier(n_neighbors=250, weights='distance')
knn.fit(X_train, Y_train)


# decision tree model
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', criterion='entropy')
dt.fit(X_train, Y_train)


# logistic regression model
lgr = LogisticRegression(C=0.05, class_weight='balanced', solver='liblinear')
lgr.fit(X_train, Y_train)


# neural network model
nn = build_model(X_train.shape[1], 167,0.001,0.0)
nn.fit(X_train, Y_train, epochs=20, batch_size=10000)
## --------------------------------------------------------------- model analysis -----------------------------------
'''
evaluate_model(knn, X_test, Y_test, 'knn')

evaluate_model(dt, X_test, Y_test, 'decision tree')

evaluate_model(lgr, X_test, Y_test, 'logistic regression')

evaluate_model(nn, X_test, Y_test, 'Neural Network')
'''
## --------------------------------------------------------------- assemble models -----------------------------------

knn_prediction = knn.predict_proba(df_test)[:,1]

dt_prediction = dt.predict_proba(df_test)[:,1]

lgr_prediction = lgr.predict_proba(df_test)[:,1]

nn_prediction = get_CV_prediction(df_train_x,df_train_y,{"lr":0.001,"dropout_rate":0},df_test,167)

target = knn_prediction * 0.25 + dt_prediction * 0.25 + lgr_prediction * 0.25 + nn_prediction * 0.25


## --------------------------------------------------------------- Final result/ submission --------------------------

submission = loadData('sample_submission.csv')
submission['TARGET'] = target
submission.to_csv('submission_final.csv', index=False)
