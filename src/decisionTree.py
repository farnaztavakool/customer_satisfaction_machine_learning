import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def loadData(path):
    df= pd.read_csv(path)
    return pd.DataFrame(df)


def train_decisionTree_model():
    # read the data
    df_train_x = loadData('X_train.csv')
    y_train = np.ravel(loadData('Y_train.csv'))
    df_test = loadData('X_test.csv')
    
    # fit the data into the model
    decisionTree = DecisionTreeClassifier(max_depth=20)
    decisionTree.fit(df_train_x, y_train)
    
    submission = loadData('sample_submission.csv')
    target = decisionTree.predict_proba(df_test)
    submission['TARGET'] = target[:,1]
    submission.to_csv('submission_decisionTree.csv', index=False)
    
    # save the model
    joblib.dump(decisionTree, 'model_decisionTree.joblib')
    return

def find_best_depth():
    # read the data
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
    
    #loss_list = []
    accuracy_list=[]
    
    # fit the data into the model
    depth_grid = range(1, 100)
    for i in depth_grid:
        decisionTree = DecisionTreeClassifier(max_depth=i)
        decisionTree.fit(X_train, Y_train)
    
        #prediction = decisionTree.predict_proba(X_test)
        #loss = log_loss(Y_test, prediction[:, 1])
        #print("loss: ", loss, "  depth: ", i)
        #loss_list.append(loss)
        
        prediction = decisionTree.predict(X_test)
        accuracy = accuracy_score(Y_test, prediction)
        print("accuracy: ", accuracy, "  depth: ", i)
        accuracy_list.append(accuracy)
    
    plt.figure(figsize=(12, 6))
    plt.plot(depth_grid, accuracy_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('Accuracy')
    plt.xlabel('depth')
    plt.ylabel('Accuracy')
    plt.show()
    return

def accuracy_score(Y_true, Y_predict):
    count = 0.0
    for i in range(-1, len(Y_true)):
        if Y_true[i] == Y_predict[i]:
            count +=1
    
    return count/len(Y_true)

def test_decisionTree():
    # read the data
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
    
    # fit the data into the model
    decisionTree = DecisionTreeClassifier(max_depth=20)
    decisionTree.fit(X_train, Y_train)
    
    plot_confusion_matrix(decisionTree, X_test, Y_test)
    plt.show()
    
    return

test_decisionTree()