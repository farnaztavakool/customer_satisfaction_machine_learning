import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss, plot_confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from data_preprocess import oversampling_dataset
import matplotlib.pyplot as plt

def loadData(path):
    df= pd.read_csv(path)
    return pd.DataFrame(df)


def train_decisionTree_model():
    # read the data
    df_train_x = loadData('X_train.csv')
    y_train = np.ravel(loadData('Y_train.csv'))
    df_test = loadData('X_test.csv')
    
    # fit the data into the model
    decisionTree = DecisionTreeClassifier(max_depth=7)
    decisionTree.fit(df_train_x, y_train)
    
    submission = loadData('sample_submission.csv')
    target = decisionTree.predict_proba(df_test)
    submission['TARGET'] = target[:,1]
    submission.to_csv('submission_decisionTree.csv', index=False)
    
    # save the model
    joblib.dump(decisionTree, 'model_decisionTree.joblib')
    return

def find_best_depth():
    # read the data
    df_train_x = loadData('X_train.csv')
    y_train = np.ravel(loadData('Y_train.csv'))
    df_test = loadData('X_test.csv')
    
    # split train data into two separate sets
    # one for training and the other one for testing
    # X_train, X_test, Y_train, Y_test = train_test_split(df_train_x, y_train, test_size = 0.3)

    # oversample the data
    '''
    X_train.insert(X_train.shape[1], 'TARGET', Y_train)
    oversampled_train = oversampling_dataset(X_train)
    X_train = oversampled_train.drop(['TARGET'], axis=1)
    Y_train = oversampled_train['TARGET']
    
    X_test.insert(X_test.shape[1], 'TARGET', Y_test)
    oversampled_test = oversampling_dataset(X_test)
    X_test = oversampled_test.drop(['TARGET'], axis=1)
    Y_test = oversampled_test['TARGET']
    '''
    #loss_list = []
    score_list=[]
    
    df_train_x = df_train_x.to_numpy()
    
    # fit the data into the model
    depth_grid = range(1, 50)
    kfold = KFold(n_splits=5)
    for i in depth_grid:
        roc_score = 0
        for train, test in kfold.split(df_train_x, y_train):
            decisionTree = DecisionTreeClassifier(max_depth=i)
            decisionTree.fit(df_train_x[train], y_train[train])
            
            prediction = decisionTree.predict_proba(df_train_x[test])[:,1]
            roc_score += roc_auc_score(y_train[test], prediction)
        print("accuracy: ", roc_score/5, "  depth: ", i)
        score_list.append(roc_score/5)
    
    plt.figure(figsize=(12, 6))
    plt.plot(depth_grid, score_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title("ROC score 5-fold CV with 'gini'")
    plt.xlabel('depth')
    plt.ylabel('ROC score')
    plt.show()
    
    score_list = []
    for i in depth_grid:
        roc_score = 0
        for train, test in kfold.split(df_train_x, y_train):
            decisionTree = DecisionTreeClassifier(max_depth=i, criterion='entropy')
            decisionTree.fit(df_train_x[train], y_train[train])
            
            prediction = decisionTree.predict_proba(df_train_x[test])[:,1]
            roc_score += roc_auc_score(y_train[test], prediction)
        print("accuracy: ", roc_score/5, "  depth: ", i)
        score_list.append(roc_score/5)
    
    plt.figure(figsize=(12, 6))
    plt.plot(depth_grid, score_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title("ROC score 5-fold CV with 'entropy'")
    plt.xlabel('depth')
    plt.ylabel('ROC score')
    # plt.show()
    plt.savefig('images/tree_score.png', bbox_inches='tight')
    return



def test_decisionTree():
    # read the data
    df_train_x = loadData('X_train.csv')
    y_train = np.ravel(loadData('Y_train.csv'))
    df_test = loadData('X_test.csv')
    
    # split train data into two separate sets
    # one for training and the other one for testing
    X_train, X_test, Y_train, Y_test = train_test_split(df_train_x, y_train, test_size = 0.3)
    
    # oversample the data
    X_train.insert(X_train.shape[1], 'TARGET', Y_train)
    oversampled_train = oversampling_dataset(X_train)
    X_train = oversampled_train.drop(['TARGET'], axis=1)
    Y_train = oversampled_train['TARGET']
    
    # fit the data into the model
    decisionTree = DecisionTreeClassifier(max_depth=20)
    decisionTree.fit(X_train, Y_train)
    
    prediction = decisionTree.predict_proba(X_test)[:,1]
    false_positive_rate, true_positive_rate, threshold1 = roc_curve(Y_test, prediction)
    print('roc_auc_score for DecisionTree: ', roc_auc_score(Y_test, prediction))
    
    plt.title('ROC - DecisionTree')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    plot_confusion_matrix(decisionTree, X_test, Y_test)
    plt.show()
    
    return

def depth_tuning(X_validation, Y_validation):
    X_validation = X_validation.to_numpy()
    
    score_list=[]
    # fit the data into the model
    depth_grid = range(1, 50)
    kfold = KFold(n_splits=5)
    for i in depth_grid:
        roc_score = 0
        for train, test in kfold.split(X_validation, Y_validation):
            decisionTree = DecisionTreeClassifier(max_depth=i, class_weight='balanced')
            decisionTree.fit(X_validation[train], Y_validation[train])
            
            prediction = decisionTree.predict_proba(X_validation[test])[:,1]
            roc_score += roc_auc_score(Y_validation[test], prediction)
        print("accuracy: ", roc_score/5, "  depth: ", i)
        score_list.append(roc_score/5)
    
    plt.figure(figsize=(12, 6))
    plt.plot(depth_grid, score_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title("ROC score 5-fold CV with 'gini' balanced")
    plt.xlabel('depth')
    plt.ylabel('ROC score')
    plt.show()
    
    score_list = []
    for i in depth_grid:
        roc_score = 0
        for train, test in kfold.split(X_validation, Y_validation):
            decisionTree = DecisionTreeClassifier(max_depth=i, criterion='entropy', class_weight='balanced')
            decisionTree.fit(X_validation[train], Y_validation[train])
            
            prediction = decisionTree.predict_proba(X_validation[test])[:,1]
            roc_score += roc_auc_score(Y_validation[test], prediction)
        print("accuracy: ", roc_score/5, "  depth: ", i)
        score_list.append(roc_score/5)
    
    plt.figure(figsize=(12, 6))
    plt.plot(depth_grid, score_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title("ROC score 5-fold CV with 'entropy' balanced")
    plt.xlabel('depth')
    plt.ylabel('ROC score')
    plt.show()
    return

