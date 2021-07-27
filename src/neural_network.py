import numpy as np
import pandas as pd
# possible change to pytorch  
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import DropCorrelatedFeatures
# possible change to pytorch  
from sklearn.neural_network import MLPClassifier
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

    
    output_dim = findBestOutputSize(df_train_x, y_train)
    
    model = build_model(df_train_x.shape[1], output_dim)
    model.fit(df_train_x, y_train)
    prediction = model.predict_proba(df_test)[:,0]
    
    model2 = build_model(X_train.shape[1], output_dim)
    model2.fit(X_train, Y_train)
    prediction2 = model.predict(X_test)[:,0]

    loss = log_loss(Y_test, prediction2)
    # print(pd.DataFrame({'ID':ID_test, "Target": prediction}))
    print('loss: ', loss)

    return pd.DataFrame({'ID':ID_test, "Target": prediction})


def loadData(path):
    df= pd.read_csv(path)
    return pd.DataFrame(df)


def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(output_dim,input_dim = input_dim, kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(1,input_dim = output_dim, kernel_initializer='uniform', activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')   
    return model

# use cross fold to get the accuracy of the model with different size 
def findBestOutputSize(data, y):
    data = data.to_numpy()
    y = y.to_numpy()
    scores = [0] * 10
    alpha_list = [i+1 for i in range(10)]
    kfold = KFold(n_splits=10)
    for alpha in alpha_list:
        for train, test in kfold.split(data, y):
            output_dim = getNumberOfNeurons(data, alpha)
            model = build_model(data[train].shape[1],output_dim)
            model.fit(data[train], y[train])
            scores[alpha-1]+=model.evaluate(data[test], y[test], verbose=0)
        scores[alpha-1] = scores[alpha-1]/10
    alpha = scores.index(min(scores)) +1
    return getNumberOfNeurons(data, alpha)

def getNumberOfNeurons(data, alpha ):
    return (data.shape[0]/(alpha*(data.shape[1]+1)))




train_NN_model().to_csv("nn_output.csv",index=False)