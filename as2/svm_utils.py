import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from data_utils import loaddata

C_range = [ 4**i for i in range(-3,8) ]
gamma_ramge = [ 4**i for i in range(-7,0) ]
kernel='rbf'
CV_FOLDS = 2

X_tr,y_tr,X_te,y_te = loaddata("test.csv")

eval_results = []
for C in C_range:
    for gamma in gamma_ramge:
        print C, gamma, kernel
        clf = SVC(C=C, gamma=gamma, kernel=kernel)
        # compute cross validation score using number of folds = CV_FOLDS
        cv_score = cross_val_score(clf, X_tr,y_tr, cv=CV_FOLDS) 
        print cv_score, np.average(cv_score)
        
        y_pred = clf.predict(X_te)
        test_score = accuracy_score(y_te, y_pred)
        
        print "Testing Accuracy", test_score
        
        eval_results.append([C, gamma,cv_score,test_score ])
        
        