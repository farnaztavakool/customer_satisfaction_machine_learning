import joblib
import numpy as np
import pandas as pd


def loadData(path):
    df= pd.read_csv(path)
    return pd.DataFrame(df)

def essemble_model():
    df_train_x = loadData('X_train.csv')
    y_train = np.ravel(loadData('Y_train.csv'))
    df_test = loadData('X_test.csv')
    
    knn = joblib.load('model_KNN.joblib')
    decisionTree = joblib.load('model_decisionTree.joblib')
    
    
    decisionTree_prediction = decisionTree.predict_proba(df_test)[:,1]
    knn_prediction = knn.predict_proba(df_test)[:,1]
    
    
    nn_prediction = np.array(loadData('NN_output.csv'))
    nn_prediction = nn_prediction[:,1]
    
    final_prediction = knn_prediction/3 + decisionTree_prediction/3 + nn_prediction/3
    
    submission = loadData('sample_submission.csv')
    submission['TARGET'] = final_prediction
    submission.to_csv('submission_essemble.csv', index=False)
    
essemble_model()