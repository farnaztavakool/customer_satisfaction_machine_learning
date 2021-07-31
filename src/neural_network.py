from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import customer_satisfaction as cs
import math
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, roc_curve
from data_preprocess import consistent_sampling
from keras.optimizers import Adam
from keras.layers import Dropout

# These variales will be set during modelling
output_dim_const = 0
input_dim_const = 0 
epoch_const = 20
batch_size_const = 10000
learn_rate_const = 0.1
dropout_const = 0



def main():
  
    df_train_x = cs.loadData('X_train.csv')
    y_train = np.ravel(cs.loadData('Y_train.csv'))
    df_test = cs.loadData('X_test.csv')
    
    print("Sampling the data for NN")
    df_train_x.insert(df_train_x.shape[1], 'TARGET', y_train)
    X_train, X_test, Y_train, Y_test = consistent_sampling(df_train_x)
  
    df_train_x = df_train_x.drop(['TARGET'], axis=1)
  
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
    dropout_rate = [0.0,0.1,0.2]
    learn_rate = [0.001,0.01,0.1] 
    model = KerasClassifier(build_fn=create_model,verbose=0,epochs=epoch_const, batch_size=batch_size_const)
    param_grid = dict( lr=learn_rate,dropout_rate=dropout_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)
    return grid.fit(x, y).best_params_
    
    
# wrapper function for gridsearch
def create_model(lr,dropout_rate):
    return build_model(input_dim_const,output_dim_const,lr, dropout_rate)

# writting the output to CSV
def write_data(prediction):
    submission = cs.loadData('sample_submission.csv')
    submission['TARGET'] = prediction
    submission.to_csv("NN_output.csv", index=False)
 
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


# do train_test split to get an estimation of the loss    
def test_nn_model(x_train,y_train,x_test,y_test,output_dim):
    
    model = build_model(x_train.shape[1],output_dim_const,learn_rate_const, dropout_const)
    model.fit(x_train,y_train,epochs=epoch_const,batch_size=batch_size_const)
    y_test_prediction = model.predict(x_test)[:,0]
    return roc_auc_score(y_test,y_test_prediction)

    
# use cross fold to find the best value for number of neurons in the hidden layer
def find_best_output_size(x_train, y_train, x_test, y_test):
   
    x_train = x_train.to_numpy()
    scores = [0] * 5
    alpha_list = [i+1 for i in range(5)]
    kfold = KFold(n_splits=5)
    for alpha in alpha_list:
        for train, test in kfold.split(x_train, y_train):
            output_dim = getNumberOfNeurons(x_train.shape[0], alpha,x_train.shape[1])
            
            model = build_model(x_train[train].shape[1], output_dim)
            model.fit(x_train[train], y_train[train], epochs=epoch_const, batch_size=batch_size_const)
        
            scores[alpha-1]+=roc_auc_score(y_test,model.predict(x_test)[:,0])
            
        scores[alpha-1] = scores[alpha-1]/5
        
    alpha = scores.index(min(scores)) +1
    return getNumberOfNeurons(x_train.shape[0], alpha, x_train.shape[1])

# use cross fold to get the final prediction
def get_CV_prediction(x, y, best_params, test_data):
    x = x.to_numpy()
    learn_rate = best_params['lr']
    dropout_const = best_params['dropout_rate']
    var = globals()
    var["learn_rate_const"] = learn_rate
    var["dropout_rate"] = dropout_const
    
    model = build_model(x.shape[1], output_dim_const,learn_rate,dropout_const)
    prediction = [0] * test_data.shape[0]
    kfold = KFold(n_splits=10)
    
    for train, test in kfold.split(x, y):
        model.fit(x[train],y[train],epochs=epoch_const,batch_size=batch_size_const)
        prediction +=model.predict(test_data)[:,0]
        
    prediction = prediction/11
    return prediction

def getNumberOfNeurons(observation_size,alpha, input_size ):
    return math.floor(observation_size/(alpha*(input_size+1)))

main()