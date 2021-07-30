import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss, plot_confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from data_preprocess import oversampling_dataset
import matplotlib.pyplot as plt


def loadData(path):
    df= pd.read_csv(path)
    return pd.DataFrame(df)


def train_lgr_model():
    # read the data
    df_train_x = loadData('X_train.csv')
    y_train = np.ravel(loadData('Y_train.csv'))
    df_test = loadData('X_test.csv')
    
    # fit the data into the model
    lgr = LogisticRegression(C=0.5, class_weight='balanced', solver='liblinear')
    lgr.fit(df_train_x, y_train)
    
    submission = loadData('sample_submission.csv')
    target = lgr.predict_proba(df_test)
    submission['TARGET'] = target[:,1]
    submission.to_csv('submission_lgr.csv', index=False)
    
    # save the model
    joblib.dump(lgr, 'model_lgr.joblib')
    return

def find_best_solver():
    # read the data
    df_train_x = loadData('X_train.csv')
    y_train = np.ravel(loadData('Y_train.csv'))
    df_test = loadData('X_test.csv')
    
    
    # split train data into two separate sets
    # one for training and the other one for testing
    # X_train, X_test, Y_train, Y_test = train_test_split(df_train_x, y_train, test_size = 0.3)
    
    df_train_x = df_train_x.to_numpy()
    
    solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    score_list = []
    kfold = KFold(n_splits=5)
    for solver in solver_list:
        roc_score = 0
        for train, test in kfold.split(df_train_x, y_train):
            # fit the data into the model
            lgr = LogisticRegression(solver=solver)
            lgr.fit(df_train_x[train], y_train[train])
            
            prediction = lgr.predict_proba(df_train_x[test])[:,1]
            roc_score += roc_auc_score(y_train[test], prediction)
        print("accuracy: ", roc_score/5, "  solver: ", solver)
        score_list.append(roc_score/5)
    
    plt.figure(figsize=(12, 6))
    plt.plot(solver_list, score_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('ROC score with 5-fold CV')
    plt.xlabel('solver')
    plt.ylabel('ROC score')
    # plt.show()
    plt.savefig('images/regression_s.png', bbox_inches='tight')
    
    return

def find_best_C():
    # read the data
    df_train_x = loadData('X_train.csv')
    y_train = np.ravel(loadData('Y_train.csv'))
    df_test = loadData('X_test.csv')
    
    
    # split train data into two separate sets
    # one for training and the other one for testing
    # X_train, X_test, Y_train, Y_test = train_test_split(df_train_x, y_train, test_size = 0.3)
    
    df_train_x = df_train_x.to_numpy()
    
    c_list = list(np.linspace(0.0001, 2, 50))
    score_list = []
    kfold = KFold(n_splits=5)
    for c in c_list:
        roc_score = 0
        for train, test in kfold.split(df_train_x, y_train):
            # fit the data into the model
            lgr = LogisticRegression(C=c, class_weight='balanced', solver='liblinear')
            lgr.fit(df_train_x[train], y_train[train])
            
            prediction = lgr.predict_proba(df_train_x[test])[:,1]
            roc_score += roc_auc_score(y_train[test], prediction)
        print("accuracy: ", roc_score/5, "  C: ", c)
        score_list.append(roc_score/5)
    
    plt.figure(figsize=(12, 6))
    plt.plot(c_list, score_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('ROC score 5-fold CV')
    plt.xlabel('c value')
    plt.ylabel('ROC score')
    # plt.show()
    plt.savefig('images/regression_c1.png', bbox_inches='tight')
    
    return

def c_value_tuning(X_validation, Y_validation):
    X_validation = X_validation.to_numpy()
    
    c_list = list(np.linspace(0.0001, 2, 50))
    score_list = []
    kfold = KFold(n_splits=5)
    for c in c_list:
        roc_score = 0
        for train, test in kfold.split(X_validation, Y_validation):
            # fit the data into the model
            lgr = LogisticRegression(C=c, class_weight='balanced', solver='liblinear')
            lgr.fit(X_validation[train], Y_validation[train])
            
            prediction = lgr.predict_proba(X_validation[test])[:,1]
            roc_score += roc_auc_score(Y_validation[test], prediction)
        print("accuracy: ", roc_score/5, "  C: ", c)
        score_list.append(roc_score/5)
    
    plt.figure(figsize=(12, 6))
    plt.plot(c_list, score_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('ROC score 5-fold CV')
    plt.xlabel('c value')
    plt.ylabel('ROC score')
    plt.show()

