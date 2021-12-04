#### Python script for every functions used
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
def load_data(file_name):
    """
    function to load the data
    """
    df = pd.read_csv(file_name +'.txt', sep=",")
    df.columns.values
    Y = df.pop(df.columns[-1]).values
    X = df.values
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
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    return clf.predict(X_test)

def precision_recall_multilabels(y_true, y_pred, labels):
    """
    Get the precision and recall of a model
    """
    recalls = np.zeros((1, 2))
    precisions = np.zeros((1, 2))
    for label in labels:
        label_array=np.full((y_true.shape),label)
        real_association = y_true == label_array
        pred_association = label_array == y_pred
        TP=np.count_nonzero(np.logical_and(real_association, pred_association))
        P=np.count_nonzero(real_association)
        FN=np.count_nonzero(pred_association)
        recalls[0,label]=TP/FN
        precisions[0,label]=TP/P
    return precisions, recalls