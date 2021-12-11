#### Python script for every functions used
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
def load_data(file_name, preprocess = False):
    """
    function to load the data
    """
    df = pd.read_csv(file_name, sep=",")
    if (preprocess):
        return pre_process(df)
    Y = df.pop(df.columns[-1]).values
    X = df
    class_labels = np.unique(Y)
    return X,Y,class_labels

def split_data(X,Y,training_size):
    """
    split the data in a training and test variables
    """
    return train_test_split(X, Y, test_size=X.shape[0]-training_size)

def predict_SVC(X_train, X_test, Y_train):
    """
    create and predict a SVC model
    """
    svc = svm.SVC()
    svc.fit(X_train, Y_train)
    return svc, svc.predict(X_test)


#def predict_Random_Forest_Classification(X_train, X_test, Y_train, depth, n_estimators):
#    class_forest = RandomForestClassifier(max_depth=depth, n_estimators=n_estimators)
#    class_forest.fit(X_train, Y_train)
#    return class_forest, pred_forest = class_forest.predict(X_test)

def precision_recall_multilabels(y_true, y_pred, labels):
    recalls = np.zeros((1, 2))
    precisions = np.zeros((1, 2))
    for label in labels:
        label_array=np.full((y_true.shape),label)
        real_association = y_true == label_array
        pred_association = label_array == y_pred
        TP=np.count_nonzero(np.logical_and(real_association, pred_association))
        P=np.count_nonzero(real_association)
        FN=np.count_nonzero(pred_association)
        if FN ==0 :
            recalls [0,label] = 1
        else:
            recalls[0,label]=TP/FN
        if P == 0 :
            precisions [0,label] = 0
        else :
            precisions[0,label]=TP/P
        

    return precisions, recalls

def find_best_depths(X,Y, n_depths=10, cvp=None, ):
    if(cvp is not None):
        # Define the cvp (cross-validation procedure) with random 1000 samples, 2/3 training size, and 1/3 test size
        cvp = ShuffleSplit(n_splits=X.shape[0], test_size=1/3, train_size=2/3, random_state=0)
    
    # Define the max depths between 1 and 10
    depths = np.linspace(1, 10, n_depths)

    # Loop on the max_depth parameter and compute negative cross entropy loss.
    tab_log_tree = []
    for i in range(n_depths):
        class_tree = DecisionTreeClassifier(max_depth=depths[i])
        tab_log_tree.append(-cross_val_score(class_tree, X, Y, scoring='neg_log_loss', cv=cvp))
    return tab_log_tree


def pre_process (dataframe):
    nodataval = '?'

    num_samples, num_features = dataframe.shape

    # Handle nodata values
    dataframe = dataframe.replace(nodataval, np.nan)

    # Convert remaining false string columns
    other_numeric_columns = ["sg", "al", "su"]
    dataframe[other_numeric_columns] = dataframe[other_numeric_columns].apply(pd.to_numeric)

    ## convert false sting to numeric    a voir
    false_strings = ["pcv", "wc", "rc"]
    for column in false_strings:
        dataframe[column]=pd.to_numeric(dataframe[column], errors='coerce')

    # Replace missing values
    fillna_mean_cols = pd.Index(
        set(dataframe.columns[dataframe.dtypes == "float64"]) - set(other_numeric_columns)
    )
    fillna_most_cols = pd.Index(
        set(dataframe.columns[dataframe.dtypes == "object"]) | set(other_numeric_columns)
    )
    dataframe[fillna_mean_cols] = dataframe[fillna_mean_cols].fillna(dataframe[fillna_mean_cols].mean())
    dataframe[fillna_most_cols] = dataframe[fillna_most_cols].fillna(dataframe[fillna_most_cols].mode().iloc[0])

    # Clear '/t' and ' yes' or ' no' values
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
    for col in cat_cols:
        dataframe[col] = dataframe[col].astype(str).apply(lambda s: s.replace('\t', ''))
        dataframe[col] = dataframe[col].astype(str).apply(lambda s: s.replace(' ', ''))


    y = dataframe.pop(dataframe.columns[-1]).values
    class_labels = np.unique(y)
    #one hot encode
    dataframe = pd.get_dummies(dataframe, drop_first=True)

    #normalization
    dataframe = (dataframe - dataframe.mean()) / (dataframe.std())
    return dataframe, y, class_labels
