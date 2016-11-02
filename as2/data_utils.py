import numpy as np
from sklearn.model_selection import train_test_split


def normalize(X_tr, X_te):
    ''' Normalize training and test data features
    Args:
        X_tr: Unnormalized training features
        X_te: Unnormalized test features
    Output:
        X_tr: Normalized training features
        X_te: Normalized test features
    '''
    X_mu = np.mean(X_tr, axis=0)
    X_tr = X_tr - X_mu
    X_sig = np.std(X_tr, axis=0)
    X_tr = X_tr/X_sig
    X_te = (X_te - X_mu)/X_sig
    return X_tr, X_te

def loaddata(filename, col_selected=None, test_size=0.2):
    ''' Load and preprocess dataset
    Args:
        filename: path to data file multimodal.csv
        col_selected: features selected
        test_size: percentage of total data to be kept for testing
    Output:
        X_tr: normalised training features
        y_tr: training labels
        X_te: normalised test features
        y_te: test labels
    '''
    
    # load all data from filename 
    with open(filename, 'r') as f:
        header_list = f.readline()
    f.close()
    header_list = header_list.split(",")
    
    # By default exclude "id column" and put all other columns
    if col_selected == None:
        col_selected = range(1, len(header_list))
    else:
        col_selected = set(col_selected)
        col_selected.add(len(header_list)-1)
        
    # Load data into X excluding headers 
    X = np.genfromtxt(filename, delimiter=",",usecols=col_selected, skip_header=1)
    
    # Shuffle data to avoid any local maxima
    np.random.shuffle(X)
    
    #last column is label, fork it out from X into y
    y = X[:,len(X[0])-1]
    X = np.delete(X, [len(X[0])-1], axis=1)
    
    X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    
    X_tr,X_te = normalize(X_tr, X_te)
    
    return X_tr,y_tr,X_te,y_te


if __name__ == '__main__':
    '''
    Example how to call loaddata
    '''
    print loaddata("test.csv")
    print loaddata("test.csv", col_selected=[1])
    print loaddata("test.csv", col_selected=[1,2])
    
    