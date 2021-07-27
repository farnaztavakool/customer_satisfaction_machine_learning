import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, chi2, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from feature_engine.selection import DropCorrelatedFeatures

def loadData(path):
    return pd.DataFrame(pd.read_csv(path))

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

def featureSelectionANOVA(train_X, train_Y, test, numOfFeatures):
    fvalue_best = SelectKBest(f_classif, k=numOfFeatures)
    fvalue_best.fit(train_X, train_Y)
    return [pd.DataFrame(fvalue_best.transform(train_X)),pd.DataFrame(fvalue_best.transform(test))]
    
def featureSelectionRFE(train_X, train_Y, test, numOfFeatures, estimator):
    rfe = RFE(estimator = estimator, n_features_to_select=numOfFeatures)
    rfe.fit(train_X, train_Y)
    return [pd.DataFrame(rfe.transform(train_X)),pd.DataFrame(rfe.transform(test))]

def featureSelectionCHI2(train_X, train_Y, test, numOfFeatures):
    new_best = SelectKBest(chi2, k=numOfFeatures)
    new_best.fit(train_X, train_Y)
    return [pd.DataFrame(new_best.transform(train_X)),pd.DataFrame(new_best.transform(test))]

def dropColumnWithName(data, name):
    column_to_drop = [c for c in data if c.startswith(name)]
    data = data.drop(columns = column_to_drop )

def normalise(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data))




def preprocessData(df_train, df_test):

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

    # use ANOVA to select k best features, the number of features is a hyperparameter which we need to test to find a best one
    num_features_ANOVA = 100
    df_train_x, df_test = featureSelectionANOVA(df_train_x, y_train, df_test, num_features_ANOVA)

    # use decisionTreeClassifier for the estimator of RFE to fun the feature selection
    estimator = DecisionTreeClassifier()
    num_features_RFE = 70
    df_train_x, df_test = featureSelectionRFE(df_train_x, y_train, df_test, num_features_RFE, estimator)


    df_train_x = normalise(df_train_x)
    df_test = normalise(df_test)

    return [df_train_x, y_train, df_test]



