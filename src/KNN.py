import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
<<<<<<< HEAD
from customer_satisfaction import preprocessData
from sklearn.metrics import log_loss
=======
>>>>>>> knn

def train_NN_model():
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
    df_train_x = loadData('X_train.csv')
    y_train = np.ravel(loadData('Y_train.csv'))
    df_test = loadData('X_test.csv')
    
    # split train data into two separate sets
    # one for training and the other one for testing
    n_rows = df_train_x.shape[0]
    rows_split = int((n_rows + 1)/2)
    X_train = df_train_x[0 : rows_split]
    Y_train = y_train[0 : rows_split]
    X_test = df_train_x[rows_split : n_rows]
    Y_test = y_train[rows_split : n_rows]

    loss_list = []
    n_neighbors_grid = range(40, 201, 40)
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
    
    
def loadData(path):
    df= pd.read_csv(path)
    return pd.DataFrame(df)

<<<<<<< HEAD
    loss = log_loss(Y_test, knn.predict_proba(X_test))
    # print(pd.DataFrame({'ID':ID_test, "Target": prediction}))
    print('loss: ', loss)
=======
train_NN_model()
>>>>>>> knn
