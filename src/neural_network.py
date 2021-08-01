from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import math
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import roc_auc_score, roc_curve
import data_preprocess as pp
from keras.optimizers import Adam
import data_preprocess as pp

# These variales will be set during modelling
output_dim_const = 0
input_dim_const = 0 
epoch_const = 20
batch_size_const = 10000
learn_rate_const = 0.1
dropout_const = 0


def main():
  
    df_train_x = pp.loadData('X_train.csv')
    y_train = np.ravel(pp.loadData('Y_train.csv'))
    df_test = pp.loadData('X_test.csv')
    
    print("Sampling the data for NN")
  
    X_train, X_test, Y_train, Y_test = train_test_split(df_train_x, y_train, test_size = 0.8)

  
    print("Finding number of neurons for NN")
    global_var = globals()
    
    output_dim = find_best_output_size(X_train, Y_train, X_test, Y_test)
    
    global_var['input_dim_const'] = df_train_x.shape[1]
    global_var['output_dim_const'] = output_dim
    
    print("gridsearch to find the best params")
    best_params = tune(X_train,Y_train)
    
    
    print("Getting the final prediction")
    prediction = get_CV_prediction(df_train_x, y_train, best_params, df_test)
  
    print("writting the data into NN_output.csv")
    write_data(prediction)
    
    print("getting the AUC socre")
    print("AUC score: ", test_nn_model(X_train, Y_train, X_test,Y_test,output_dim))


def tune(x,y):
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    learn_rate = [0.001,0.01,0.1] 
    model = KerasClassifier(build_fn=create_model,verbose=0,epochs=epoch_const, batch_size=batch_size_const)
    param_grid = dict( lr=learn_rate,dropout_rate=dropout_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)
    return grid.fit(x, y).best_params_
    
    
# wrapper function for gridsearch
def create_model(lr,dropout_rate):
    return build_model(input_dim_const,output_dim_const,lr, dropout_rate)
 
# build the NN model based on the given config   
def build_model(input_dim, output_dim,learn_rate=0.01,dropout_rate=0.0):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, kernel_initializer='uniform', activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim,activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    optimizer = Adam(learning_rate=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])   
    return model


    
# use cross fold to find the best value for number of neurons in the hidden layer
def find_best_output_size(x_train, y_train, x_test, y_test):
   
    x_train = x_train.to_numpy()
    scores = [0] * 5
    alpha_list = [i+1 for i in range(10)]
    kfold = KFold(n_splits=5)
    for alpha in alpha_list:
        for train, test in kfold.split(x_train, y_train):
            output_dim = getNumberOfNeurons(x_train.shape[0], alpha,x_train.shape[1])
            
            model = build_model(x_train[train].shape[1], output_dim)
            model.fit(x_train[train], y_train[train], epochs=epoch_const, batch_size=batch_size_const)
        
            scores[alpha-1]+=roc_auc_score(y_test,model.predict(x_test)[:,0])
            
        scores[alpha-1] = scores[alpha-1]/5
        
    alpha = scores.index(min(scores)) +1
    var = globals()
    output_dim = getNumberOfNeurons(x_train.shape[0], alpha, x_train.shape[1])
    var["output_dim_const"] = output_dim
    return output_dim

# use cross fold to get the final prediction
def get_CV_prediction(x, y, best_params, test_data, output_dim):
    x = x.to_numpy()
    learn_rate = best_params['lr']
    dropout_const = best_params['dropout_rate']
    var = globals()
    var["learn_rate_const"] = learn_rate
    var["dropout_rate"] = dropout_const
    print(learn_rate, dropout_const,output_dim)
    model = build_model(x.shape[1], output_dim,learn_rate,dropout_const)
    prediction = [0] * test_data.shape[0]
    kfold = KFold(n_splits=10)
    
    for train, test in kfold.split(x, y):
        history = model.fit(x[train],y[train],epochs=epoch_const,batch_size=batch_size_const)
        prediction +=model.predict(test_data)[:,0]
        
    prediction = prediction/11
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('images/network_prediction.png', bbox_inches='tight')
    return prediction

# get the number of nurons
def getNumberOfNeurons(observation_size,alpha, input_size ):
    return math.floor(observation_size/(alpha*(input_size+1)))
