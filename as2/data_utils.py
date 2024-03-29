import numpy as np
from sklearn.model_selection import train_test_split

def get_custom_validation_sets(X_tr, y_tr):
    '''
    cv_custom is a list of tuples (train, test)
    cv_custom_y is a list of tuples (train_y, test_y)
    '''
    cv_custom = []
    cv_custom_y = []
    for idx, val_set in enumerate(X_tr):
        tr_sets = []
        y_tr_sets = []
        for tr_set in X_tr[:idx] + X_tr[idx + 1:]:
            tr_sets += list(tr_set)
        
        for y_tr_set in y_tr[:idx] + y_tr[idx + 1:]:
            y_tr_sets += list(y_tr_set)
        
        cv_custom.append((tr_sets, list(val_set)))
        cv_custom_y.append((y_tr_sets, list(y_tr[idx])))
    
    return cv_custom, cv_custom_y

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
    if(index>len(X)):
        return X,[]
    while(index+forwardCounter+1 < len(X) and X[index+forwardCounter,0]==X[index+forwardCounter+1,0] and X[index+1-backwardCounter,0]==X[index-backwardCounter,0]):
        forwardCounter = forwardCounter + 1
        backwardCounter = backwardCounter + 1
    if(index+forwardCounter+1 >= len(X)):
        splitIndex = index+forwardCounter
        return X[0:splitIndex+1],[]
    elif(X[index+forwardCounter,0]!=X[index+forwardCounter+1,0]):
        splitIndex = index+forwardCounter
    else:
        splitIndex = index-backwardCounter
    return X[0:splitIndex+1],X[splitIndex+1:]

def loaddata(filename, test_slice = 0, split=3, shuffle=False, col_selected=None, test_size=0.25  ):
    ''' Load and preprocess dataset
    Args:
        filename: path to data file multimodal.csv
        col_selected: features selected
        test_size: percentage of total data to be kept for testing
        split = split_size 
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
        col_selected = range(0, len(header_list))
    #else:
        #col_selected.add(0)
        #col_selected.add(len(header_list)-1)
        #col_selected = set(col_selected)

    # Load data into X excluding headers 
    X = np.genfromtxt(filename, delimiter=",",usecols=col_selected, skip_header=1)
    
    
    # testing data, all folds of training data
    test_start_index = test_slice*(test_size * len(X))
    X_size = len(X)
    train_split,X = getSplitIndex(X, test_start_index)
    test_split,train_split_next = getSplitIndex(X, test_size * X_size)
    if(len(train_split)>0 and len(train_split_next)>0):
        train_split = np.concatenate((train_split, train_split_next), axis=0)
    elif(len(train_split_next)>0):
        train_split = train_split_next
    train_sample_size = len(train_split)
    
    if shuffle :
        # Shuffle training data to avoid any local maxima
        np.random.shuffle(train_split)
    
    #Normalize data
    #last column is label, fork it out from X into y
    # id is column 0
    y_te = test_split[:,len(X[0])-1]
    X_te = test_split[:,1:len(X[0])-1]
    
    #print "Total length of loaded data", len(X[0])
    #print "Length Test features -",  len(X_te[0])
    
    #last column is label, fork it out from X into y
    # id is column 0
    id_tr = train_split[:,0]
    y_tr = train_split[:,len(X[0])-1]
    X_tr = train_split[:,1:len(X[0])-1]
    
    #print "Length Training features -",  len(X_tr[0])
    
    # Normalize training and test features 
    X_tr,X_te = normalize(X_tr, X_te)
    
    # append normalized X_tr with y_tr to pass it for CV sets
    z = np.zeros((len(y_tr),1))
    z[:,0]=y_tr
    train_split=np.append(X_tr, z, axis=1)
    # append id in the start
    z = np.zeros((len(id_tr),1))
    z[:,0]=id_tr
    train_split = np.append(z,train_split, axis=1)
    
    #print "Length cols in train_split -",  len(train_split[0])
    
    # all training splits
    train_samples = []
    split_size = (train_sample_size/split)
    i = 1
    while(i<split):
        train_subsplit,train_split = getSplitIndex(train_split, split_size)
        #print len(train_subsplit),len(train_split)
        train_samples.append(train_subsplit)
        i=i+1
    train_samples.append(train_split)
    
    X_tr,y_tr = [],[]
    for fold in train_samples:
        last_col = len(fold[0])
        X_fold = fold[:,1:last_col-1]
        y_fold = fold[:,last_col-1]
        X_tr.append(X_fold)
        y_tr.append(y_fold)
        
    #    print "Features in X_fold", len(X_fold[0]), len(y_fold), len(X_fold)
        
    #print len(X_tr), len(y_tr), len(X_te), len(y_te)
    return X_tr,y_tr,X_te,y_te    


if __name__ == '__main__':
    '''
    Example how to call loaddata
    '''
    loaddata("output-feature-engineering-multimodal.csv")
#    print loaddata("test.csv", col_selected=[1])
#    print loaddata("test.csv", col_selected=[1,2])
    
    