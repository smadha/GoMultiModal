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

def getSplitIndex(X,index):
    forwardCounter = 0
    backwardCounter = 0
    while(X[index+forwardCounter,0]==X[index+forwardCounter+1,0] and X[index+1-backwardCounter,0]==X[index-backwardCounter,0]):
        forwardCounter = forwardCounter + 1
        backwardCounter = backwardCounter + 1
    if(X[index+forwardCounter,0]!=X[index+forwardCounter+1,0]):
        splitIndex = index+forwardCounter
    else:
        splitIndex = index-backwardCounter
    return X[0:splitIndex],X[splitIndex+1:]

def loaddata(filename, col_selected=None, test_size=0.25, split=4):
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
    #np.random.shuffle(X)

    test_split,train_split = getSplitIndex(X, test_size * len(X))
    train_sample_size = len(train_split)
    print len(test_split),len(train_split)
    train_samples = []
    split_size = (train_sample_size/split)
    i = 1
    while(i<split):
        train_subsplit,train_split = getSplitIndex(train_split, split_size)
        print len(train_subsplit),len(train_split)
        train_samples.append(train_subsplit)
        i=i+1
    train_samples.append(train_split)
    #last column is label, fork it out from X into y
    #y_te = test_split[:,len(X[0])-1]
    #X_te = test_split[:,0:len(X[0])-2]
    #X = np.delete(X, [len(X[0])-1], axis=1)
    
    #X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    
    #X_tr,X_te = normalize(X_tr, X_te)
    
    return test_split,train_samples


if __name__ == '__main__':
    '''
    Example how to call loaddata
    '''
    loaddata("output-feature-engineering-multimodal.csv")
#    print loaddata("test.csv", col_selected=[1])
#    print loaddata("test.csv", col_selected=[1,2])
    
    