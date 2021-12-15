#### Python script for every functions used
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm
import numpy as np
from sklearn.metrics import  confusion_matrix, classification_report, plot_confusion_matrix, recall_score,  roc_auc_score, log_loss, f1_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

def load_data(file_name, preprocess = False, columns_name = None):
    """
    Code by : Dylan

    Summary
    This function load the data and separate the labels and the features

    Parameters
    ----------
    param1 : string
        Name of the file required
    param2 : Boolean (Optional)
        Allow pre_processing or not (by Default disabled)

    Returns
    -------
    pd.Dataframe  : 
        Dataframe of the different features
    np.array :
        The one-hot encoded label
    np.array :
        The different unique labels

    """
    df = pd.read_csv(file_name, sep=",", names=columns_name)
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

def predict_SVC(X_train, X_test, Y_train, kernel='linear'):
    """
    Code by : Dylan Rachwal

    Summary
    create and predict using SVM algorithm

    Parameters
    ----------
    param1 : pd.Dataframe
        Dataset for training
    param2 : pd.Dataframe
        Dataset for testing
    param3 : np.array
        labels for training
    param4 : string
        Kernel for the SVM. Can be 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    Returns
    -------
    sklearn.svm.SVC : 
        model of the SVM
    
    np.array :
        predicted labels of the test dataset

    """
    svc = svm.SVC(kernel = kernel)
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
    """
    Code by : Dylan

    Summary
    This function is used for the pre processing of the dataset

    Parameters
    ----------
    param1 : pd.Dataframe
        DataFrame which is cleaned and normalized
    Returns
    -------
    pd.Dataframe  : 
        Dataframe of the different features
    np.array :
        The one-hot encoded label
    np.array :
        The different unique labels

    """
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
    
    #randomize order
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    
    dataframe = pd.get_dummies(dataframe, drop_first=True)
    y = dataframe.pop(dataframe.columns[-1]).values
    class_labels = np.unique(y)
    #one hot encode
    

    return dataframe, y, class_labels

def min_max_normalization(dataframe):
    """
    Code by : Alexandre Thouvenot

    Summary
    Compute KNN algorithm

    Parameters
    ----------
    param1 : pd.Dataframe
        DataFrame
    Returns
    -------
    pd.Dataframe 
        Normalised dataframe

    """
    return (dataframe - dataframe.min()) / (dataframe.max()- dataframe.min())

class PCA_dec:
    """
    Code by : Alexandre Thouvenot

    Summary
    Class used to compute PCA decomosition with EVD method

    """
    def __init__(self, data):
        self.data = data
        self.cov_matrix = np.cov(np.transpose(self.data))
        self.eig_val, self.eig_vect = np.linalg.eig(self.cov_matrix)
    
    def exp_variance(self):
        return np.cumsum(self.eig_val) / sum(self.eig_val)
    
    def PCA_decomposition(self, dim):
        reduce_data = np.dot(np.transpose(self.eig_vect[:,:dim]), np.transpose(self.data))
        return np.transpose(reduce_data)
    
def KNN(X_train,Y_train,X_test,K):
    """
    Code by : Alexandre Thouvenot

    Summary
    Compute KNN algorithm

    Parameters
    ----------
    param1 : np.Array
        Array which contains training values
    param2 : np.Array
        Array which contains training values labels
    param3 : np.Array
        Array which contains test values
    param4 : Int
        Number of neighbors
    Returns
    -------
    List : 
        List of index of training blocks
    List :
        List of index of test blocks

    """
    Y_test = []
    for x in X_test:
        index_list = np.argsort(np.linalg.norm((X_train-x), axis=1))[:K]
        label_list = Y_train[index_list]
        val, nb = np.unique(label_list, return_counts=True)
        Y_test.append(val[np.argmax(nb)])
        
    return np.array(Y_test)

def K_Fold(X, n_split):
    """
    Code by : Alexandre Thouvenot

    Summary
    Split data into n_split block to compute KFold cross validation

    Parameters
    ----------
    param1 : np.Array
        Array which contains training values
    param2 : Int
        Number of block
    Returns
    -------
    List : 
        List of index of training blocks
    List :
        List of index of test blocks

    """
    n_data, _ = X.shape
    train_index_list = []
    test_index_list = []
    bloc_size = n_data // n_split
    for i in range(0, n_split):
        bloc_test = np.array(range(i * bloc_size, (i+1) * bloc_size))
        bloc_train = np.delete(np.array(range(0, bloc_size * n_split)), bloc_test)
        train_index_list.append(bloc_train)
        test_index_list.append(bloc_test)
        
    return train_index_list, test_index_list

def display_data_histogram(X, x_plot, y_plot):
    """
    Code by : Alexandre Thouvenot

    Summary
    Plot dataframe features histogram

    Parameters
    ----------
    param1 : pd.Dataframe
        DataFrame
    param2 : Int
        x size of subplot
    param2 : Int
        y size of subplot
    Returns
    -------
    plt.axs : 
        Figure of multiple histogram

    """
    fig, axs = plt.subplots(x_plot, y_plot)
    for i, column in enumerate(X.columns):
        axs[i//x_plot,i%y_plot].hist(X[column].values, bins=20)
        axs[i//x_plot,i%y_plot].set_title('Column ' + column)
    fig.tight_layout()
    return axs
    

def display_confusion_matrix(model,X_test,Y_test,Y_pred):
    """
    Code by : Dylan Rachwal

    Summary
    Plot and return the confusion matrix

    Parameters
    ----------
    param1 : sklearn.model
        model used to predict values
    param2 : pd.Dataframe
        Dataset for testing
    param3 : np.array
        labels for testing
    param4 : np.array
        labels predicted    
    Returns
    -------
    np.array : 
        Confusion_matrix of the model

    """
    confusion_matrix_test=confusion_matrix(Y_test,Y_pred)
    plot_confusion_matrix(model,X_test,Y_test)
    return confusion_matrix_test

def multiple_prediction_scores(model, X, Y, cv, scoring):
    """
    Code by : Dylan Rachwal

    Summary
    Return the wanted scores of the model using cross validation

    Parameters
    ----------
    param1 : sklearn.model
        model used to predict values
    param2 : pd.Dataframe
        Dataset 
    param3 : np.array
        labels 
    param4 : Int
        size of the cross validation 
    param5 : List[str]
        List of the scores
    Returns
    -------
    np.array : 
        Confusion_matrix of the model

    """
    return cross_validate(model, X, Y, cv = cv,  scoring=scoring)    
    
    