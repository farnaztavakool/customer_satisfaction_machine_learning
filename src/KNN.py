import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
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
    y_train = np.ravel(loadData('Y_train.csv'))
    df_test = cs.loadData('X_test.csv')
    
    # split train data into two separate sets
    # one for training and the other one for testing
    n_rows = df_train_x.shape[0]
    rows_split = int((n_rows + 1)/2)
    X_train = df_train_x[0 : rows_split]
    Y_train = y_train[0 : rows_split]
    X_test = df_train_x[rows_split : n_rows]
    Y_test = y_train[rows_split : n_rows]

    loss_list = []
    n_neighbors_grid = range(8, 41, 8)
    for i in n_neighbors_grid:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, Y_train)
        loss = log_loss(Y_test, knn.predict_proba(X_test))
        loss_list.append(loss)
        print('loss: ', loss, "  number of neighbours: ", i)
        
    plt.figure(figsize=(12, 6))
    plt.plot(n_neighbors_grid, loss_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('Loss')
    plt.xlabel('number of neighbors')
    plt.ylabel('Loss')
    plt.show()

def train_KNN_model():
    df_train_x = cs.loadData('X_train.csv')
    y_train = np.ravel(cs.loadData('Y_train.csv'))
    df_test = cs.loadData('X_test.csv')
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(df_train_x, y_train)
    
    submission = cs.loadData('sample_submission.csv')
    target = knn.predict(df_test)
    submission['TARGET'] = target
    submission.to_csv('submission.csv', index=False)
    
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
    
    
    
train_KNN_model()
