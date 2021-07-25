from sklearn.neighbors import KNeighborsClassifier
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

    n_neighbors = 100
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train)

    loss = log_loss(Y_test, knn.predict_proba(X_test))
    # print(pd.DataFrame({'ID':ID_test, "Target": prediction}))
    print('loss: ', loss)