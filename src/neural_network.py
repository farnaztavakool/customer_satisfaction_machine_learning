import numpy as np
import pandas as pd
# possible change to pytorch  
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Dense

from customer_satisfaction import preprocessData

def train_NN_model():
    df_train = loadData("train.csv")
    df_test = loadData("test.csv")

    # dropping "ID" from both training and test data
    ID_train = df_train["ID"].copy()
    ID_test = df_test["ID"].copy()
    df_train = df_train.drop(columns = "ID")
    df_test = df_test.drop(columns = "ID")

    df_train_x, y_train, df_test = preprocessData(df_train, df_test)
    
    # split train data into two separate sets
    # one for training and the other one for testing
    n_rows = df_train_x.shape[0]
    rows_split = int((n_rows + 1)/2)
    X_train = df_train_x[0 : rows_split]
    Y_train = y_train[0 : rows_split]
    X_test = df_train_x[rows_split : n_rows]
    Y_test = y_train[rows_split : n_rows]

    model = build_model(X_train)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)[:,0]

    loss = log_loss(Y_test, prediction)
    # print(pd.DataFrame({'ID':ID_test, "Target": prediction}))
    print('loss: ', loss)






def loadData(path):
    df= pd.read_csv(path)
    return pd.DataFrame(df)


def build_model(data):
    model = Sequential()
    model.add(Dense(120,input_dim = data.shape[1], kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(1,input_dim = 120, kernel_initializer='uniform', activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')   
    return model


train_NN_model()