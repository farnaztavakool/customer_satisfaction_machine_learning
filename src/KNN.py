import numpy as np
import matplotlib.pyplot as plt
#from sklearn.externals import joblib
import joblib
from sklearn.metrics import plot_confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split
# from sklearn.model_selection import train_test_split
from data_preprocess import consistent_sampling
# undersampling_dataset, 
import customer_satisfaction as cs

def find_best_k_value():
    df_train_x = cs.loadData('X_train.csv')
    y_train = np.ravel(cs.loadData('Y_train.csv'))
    df_test = cs.loadData('X_test.csv')
    
    # split train data into two separate sets
    # one for training and the other one for testing
    X_train, X_test, Y_train, Y_test = train_test_split(df_train_x, y_train, test_size = 0.8)
    # df_train_x.insert(df_train_x.shape[1], 'TARGET', y_train)
    # X_train, X_test, Y_train, Y_test = consistent_sampling(df_train_x)
    
    # # undersample the data
    # X_train.insert(X_train.shape[1], 'TARGET', Y_train)
    # undersampled_train = undersampling_dataset(X_train)
    # X_train = undersampled_train.drop(['TARGET'], axis=1)
    # Y_train = undersampled_train['TARGET']


    X_train = X_train.to_numpy()
    score_list = []
    n_neighbors_grid = range(200, 1001, 200)
    kfold = KFold(n_splits=5)
    for i in n_neighbors_grid:
        score = 0
        for train, test in kfold.split(X_train, Y_train):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train[train], Y_train[train])
            score += roc_auc_score(Y_train[test], knn.predict_proba(X_train[test])[:,1])
        
        score_list.append(score/5)
        print('roc_auc_score: ', score/5, "  number of neighbours: ", i)
        
    plt.figure(figsize=(12, 6))
    plt.plot(n_neighbors_grid, score_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('roc_auc_score with 5-fold CV')
    plt.xlabel('number of neighbors')
    plt.ylabel('score')
    plt.show()

def train_KNN_model():
    df_train_x = cs.loadData('X_train.csv')
    y_train = np.ravel(cs.loadData('Y_train.csv'))
    df_test = cs.loadData('X_test.csv')
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(df_train_x, y_train)
    
    submission = cs.loadData('sample_submission.csv')
    target = knn.predict_proba(df_test)
    submission['TARGET'] = target[:,1]
    submission.to_csv('submission_KNN.csv', index=False)
    
    # save the model
    joblib.dump(knn, 'model_KNN.joblib')
    
def K_value_tuning(X_validation, Y_validation):
    X_validation = X_validation.to_numpy()
    score_list = []
    n_neighbors_grid = range(20, 1001, 20)
    kfold = KFold(n_splits=5)
    for i in n_neighbors_grid:
        score = 0
        for train, test in kfold.split(X_validation, Y_validation):
            knn = KNeighborsClassifier(n_neighbors=i, weights='distance')
            knn.fit(X_validation[train], Y_validation[train])
            score += roc_auc_score(Y_validation[test], knn.predict_proba(X_validation[test])[:,1])
        
        score_list.append(score/5)
        print('roc_auc_score: ', score/5, "  number of neighbours: ", i)
        
    plt.figure(figsize=(12, 6))
    plt.plot(n_neighbors_grid, score_list, color='red', linestyle='dashed', marker='o',
            markerfacecolor='blue', markersize=10)
    plt.title('roc_auc_score with 5-fold CV')
    plt.xlabel('number of neighbors')
    plt.ylabel('score')
    plt.show()
