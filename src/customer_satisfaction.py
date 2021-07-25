import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import DropCorrelatedFeatures
# possible change to pytorch  
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense

def loadData(path):
    df= pd.read_csv(path)
    return pd.DataFrame(df)

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
    
def dropColumnWithName(data, name):
    column_to_drop = [c for c in data if c.startswith(name)]
    data = data.drop(columns = column_to_drop )

def normalise(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data))
  
def build_model(data):
    model = Sequential()
    model.add(Dense(120,input_dim = data.shape[1], kernel_initializer='uniform', activation = 'tanh'))
    model.add(Dense(1,input_dim = 120, kernel_initializer='uniform', activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')   
    return model


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

y_train = df_train['TARGET'].copy()
df_train_x = df_train.drop(columns="TARGET")

quasi_res = quasiConstantRemoval(df_train_x, 0.01, df_test)
df_train_x = quasi_res[0]
df_test = quasi_res[1]

correlated  = dropCorrelatedFeatures(df_train_x,df_test)
df_train_x = correlated[0]
df_test = correlated[1]


df_train_x = normalise(df_train_x)
df_test = normalise(df_test)

model = build_model(df_train_x)
model.fit(df_train_x, y_train)
prediction = model.predict(df_test)[:,0]

print(pd.DataFrame({'ID':ID_test, "Target": prediction}))
