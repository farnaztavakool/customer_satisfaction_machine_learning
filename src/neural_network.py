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
from sklearn.metrics import roc_auc_score, roc_curve
from data_preprocess import consistent_sampling
# These variales will be set during modelling
output_dim_const = 0
input_dim_const = 0 
epoch_const = 0
batch_size_const = 0



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

 

'''
# Tuning learning rate
-------(1)-----------
def step_decay_schedule(init_lr = 1e-3, decay_factor = 0.75, step_size = 10):
    #Wrapper to create learning rate scheduler with step decay schedule
    def schedule(epochs):
        return init_lr * (decay_factor ** np.floor(epochs/step_size))
    return LearningRateScheduler(schedule)
lr_schedule = step_decay_schedule(init_lr=1e-4, decay_factor=0.72, step_size=2)


'''


# Learning rate schedule (2)
initial_learning_rate = 0.01
epochs = 100
decay = initial_learning_rate / epochs

def lr_time_based(epoch, lr):
    return lr * 1 / (1 + decay * epoch)



def tune(x,y):
    epochs = [10,20,30] 
    batch_size = [1000,5000,10000]
    
   
    #learn_rate = [0.0005, 0.001, 0.00146]
    
    model = KerasClassifier(build_fn=create_model,verbose=0)
    param_grid = dict(epochs=epochs,batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)
    return grid.fit(x, y).best_params_
    
    
# wrapper function for gridsearch
def create_model():
    return build_model(input_dim_const,output_dim_const)

# writting the output to CSV
def write_data(prediction):
    submission = cs.loadData('sample_submission.csv')
    submission['TARGET'] = prediction
    submission.to_csv("NN_output.csv", index=False)
 
# build the NN model based on the given config   
def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(output_dim,input_dim = input_dim, kernel_initializer='uniform', activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   
    return model


# fit the NN model 
def fit_model(x,y,output_dim,test,epoch,batch_size):
    model = build_model(x.shape[1],output_dim)
    model.fit(x, y,epochs=epoch,batch_size=batch_size,verbose=1)
    
    # Can be used for (1)
    # model.fit(x, y,epochs=epoch,batch_size=batch_size,verbose=1, callbacks=[lr_schedule]) 
    
    # model.fit(x, y,epochs=20,batch_size=10000)
    
    # Can be used for (2)
    model.fit(x, y,epochs=epoch,batch_size=batch_size,verbose=1, callbacks=[LearningRateScheduler(lr_time_based, verbose=1)])
    return model.predict(test)[:,0]

# do train_test split to get an estimation of the loss    
def test_nn_model(x,y,output_dim):
    n_rows = x.shape[0]
    rows_split = int((n_rows + 1)/2)
    x_train = x[0 : rows_split]
    y_train = y[0 : rows_split]
    x_test = x[rows_split : n_rows]
    y_test = y[rows_split : n_rows]
    y_test_prediction = fit_model(x_train,y_train,output_dim, x_test,epoch_const,batch_size_const)
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
            model.fit(x_train[train], y_train[train])
        
            scores[alpha-1]+=roc_auc_score(y_test,model.predict(x_test)[:,0])
            
        scores[alpha-1] = scores[alpha-1]/5
        
    plt.figure(figsize=(12, 6))
    plt.plot(alpha_list, scores, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=5)
    plt.title('roc_auc_score')
    plt.ylabel('score')
    plt.show(block=False)
    alpha = scores.index(min(scores)) +1
    return getNumberOfNeurons(x_train.shape[0], alpha, x_train.shape[1])

# use cross fold to get the final prediction
def get_CV_prediction(x, y, best_params, test_data):
    x = x.to_numpy()

    epochs = best_params['epochs']
    batch_size = best_params['batch_size']
    
    var = globals()
    var["epoch_const"] = epochs
    var["batch_size_const"] = batch_size

    prediction = fit_model(x,y,output_dim_const, test_data,epochs,batch_size)
    
    kfold = KFold(n_splits=10)
    
    for train, test in kfold.split(x, y):
        prediction+= fit_model(x[train],y[train],output_dim_const,test_data,epochs,batch_size)
        
    prediction = prediction/11
    return prediction

def getNumberOfNeurons(observation_size,alpha, input_size ):
    return math.floor(observation_size/(alpha*(input_size+1)))

main()