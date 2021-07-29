import numpy as np
import matplotlib.pyplot as plt
#from sklearn.externals import joblib
import joblib
from sklearn.metrics import log_loss, plot_confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from data_preprocess import undersampling_dataset
import customer_satisfaction as cs

def find_best_k_value():
    '''
    df_train = loadData("train.csv")
    df_test = loadData("test.csv")

    # dropping "ID" from both training and test data
    ID_train = df_train["ID"].copy()
    ID_test = df_test["ID"].copy()
    df_train = df_train.drop(columns = "ID")
    df_test = df_test.drop(columns = "ID")

    df_train_x, y_train, df_test = preprocessData(df_train, df_test)
    '''
    df_train_x = cs.loadData('X_train.csv')
    y_train = np.ravel(cs.loadData('Y_train.csv'))
    df_test = cs.loadData('X_test.csv')
    
    # split train data into two separate sets
    # one for training and the other one for testing
    X_train, X_test, Y_train, Y_test = train_test_split(df_train_x, y_train, test_size = 0.3)

    score_list = []
    n_neighbors_grid = range(200, 1001, 200)
    kfold = KFold(n_splits=5)
    for i in n_neighbors_grid:
        score = 0
        for x, y in kfold.split(X_train, Y_train):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(x, y)
            score += roc_auc_score(Y_test, knn.predict_proba(X_test)[:,1])
        
        score_list.append(score/5)
        print('roc_auc_score: ', score/5, "  number of neighbours: ", i)
        
    plt.figure(figsize=(12, 6))
    plt.plot(n_neighbors_grid, score_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('roc_auc_score with 5-fold CV')
    plt.xlabel('number of neighbors')
    plt.ylabel('score')
    plt.show()

def train_KNN_model():
    df_train_x = cs.loadData('X_train.csv')
    y_train = np.ravel(cs.loadData('Y_train.csv'))
    df_test = cs.loadData('X_test.csv')
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(df_train_x, y_train)
    
    submission = cs.loadData('sample_submission.csv')
    target = knn.predict_proba(df_test)
    submission['TARGET'] = target[:,1]
    submission.to_csv('submission_KNN.csv', index=False)
    
    # save the model
    joblib.dump(knn, 'model_KNN.joblib')
    
def test_KNN_model():
    df_train_x = cs.loadData('X_train.csv')
    y_train = np.ravel(cs.loadData('Y_train.csv'))
    df_test = cs.loadData('X_test.csv')
    
    # split train data into two separate sets
    # one for training and the other one for testing
    X_train = df_train_x[0 : 10000]
    Y_train = y_train[0 : 10000]
    X_test = df_train_x[10000 : 15000]
    Y_test = y_train[10000 : 15000]
    
    knn = KNeighborsClassifier(n_neighbors=400)
    knn.fit(X_train, Y_train)
    predict = knn.predict_proba(X_test)
    loss = log_loss(Y_test, predict)
    score = knn.score(X_test, Y_test)
    print(predict[:,1])
    print('loss: ', loss)
    print('score: ',score)
    
    joblib.dump(knn, 'model_KNN.joblib')
    
def test_load_model():
    df_train_x = cs.loadData('X_train.csv')
    y_train = np.ravel(cs.loadData('Y_train.csv'))
    df_test = cs.loadData('X_test.csv')
    
    # split train data into two separate sets
    # one for training and the other one for testing
    X_train = df_train_x[0 : 10000]
    Y_train = y_train[0 : 10000]
    X_test = df_train_x[10000 : 15000]
    Y_test = y_train[10000 : 15000]
    
    knn = joblib.load('model_KNN.joblib')
    predict = knn.predict_proba(X_test)
    loss = log_loss(Y_test, predict)
    score = knn.score(X_test, Y_test)
    print(predict[:,1])
    print('loss: ', loss)
    print('score: ',score)
    
find_best_k_value()
