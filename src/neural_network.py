from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import customer_satisfaction as cs
import math
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from data_preprocess import undersampling_dataset
# These variales will be set during modelling
output_dim_const = 0
input_dim_const = 0 
epoch_const = 0
batch_size_const = 0

def sample(train_x, train_y):
    X_train, X_test, Y_train, Y_test = train_test_split(train_x, y_train, test_size = 0.3)

    # undersample the data
    X_train.insert(X_train.shape[1], 'TARGET', Y_train)
    undersampled_train = undersampling_dataset(X_train)
    X_train = undersampled_train.drop(['TARGET'], axis=1)
    Y_train = undersampled_train['TARGET']
    return [X_train, Y_train, X_test, Y_test]

def main():
  
    df_train_x = cs.loadData('X_train.csv')
    y_train = np.ravel(cs.loadData('Y_train.csv'))
    df_test = cs.loadData('X_test.csv')
    
    # split train data into two separate sets
    # one for training and the other one for testing
   

    global_var = globals()
    
    sampled_data = sample(df_train_x, y_train)
    output_dim = find_best_output_size(sampled_data)
    # output_dim = 125
    # global_var['input_dim_const'] = df_train_x.shape[1]
    # global_var['output_dim_const'] = output_dim
    
 
    # print("the best number of neurons is:", output_dim_const)
   
    # prediction = get_CV_prediction(df_train_x, y_train, output_dim_const, df_test)
    # write_data(prediction)
    
    # print('roc_auc_score: ', test_nn_model(df_train_x,y_train,output_dim))

   

def tune(x,y):
    epochs = [10,20,30]
    batch_size = [1000,5000,10000]
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
    # model.fit(x, y,epochs=20,batch_size=10000)
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
def find_best_output_size(sampled_data):
    x = sampled_data[0]
    y = sampled_data[1]
    x_test = sampled_data[2]
    y_test = sampled_data[3]
    
    scores = [0] * 5
    alpha_list = [i+1 for i in range(5)]
    kfold = KFold(n_splits=5)
    for alpha in alpha_list:
        for train, test in kfold.split(x, y):
            output_dim = getNumberOfNeurons(x.shape[0], alpha,x.shape[1])
            model = fit_model(x[train],y[train],output_dim,x[test],1,1000)
            scores[alpha-1]+=roc_auc_score(y_test,model.predict_proba(x_test))
            
        scores[alpha-1] = scores[alpha-1]/5
        
    plt.figure(figsize=(12, 6))
    plt.plot(alpha_list, scores, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=5)
    plt.title('roc_auc_score')
    plt.ylabel('score')
    # plt.show(block=False)
    plt.savefig('images/network_score.png', bbox_inches='tight')
    alpha = scores.index(min(scores)) +1
    return getNumberOfNeurons(x.shape[0], alpha, x.shape[1])

# use cross fold to get the final prediction
def get_CV_prediction(x, y, output_dim, test_data):
    x = x.to_numpy()
    best_params = tune(x,y)
    epochs = best_params['epochs']
    batch_size = best_params['batch_size']
    # epochs = 20
    # batch_size = 10000
    var = globals()
    var["epoch_const"] = epochs
    var["batch_size_const"] = batch_size
    model = build_model(x.shape[1],output_dim)
    prediction = fit_model(x,y,output_dim, test_data,epochs,batch_size)
    
    kfold = KFold(n_splits=10)
    
    for train, test in kfold.split(x, y):
        prediction+= fit_model(x[train],y[train],output_dim,test_data,epochs,batch_size)
        
    prediction = prediction/11
    return prediction

def getNumberOfNeurons(observation_size,alpha, input_size ):
    return math.floor(observation_size/(alpha*(input_size+1)))

main()
