from __future__ import print_function

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from data_utils import loaddata

C_range = [ 4**i for i in range(-3,8) ]
gamma_ramge = [ 4**i for i in range(-7,0) ]
deg_range =  range(1,4)

# Number of folds in Cross validation
CV_FOLDS = 5
# Number of parallel jobs
parallel = 2


def report_svm(X_tr,y_tr,X_te,y_te, print_CV_report = False):
    svr = SVC()
    
    parameters = [{ 'kernel':['poly'], 'C':C_range, 'degree':deg_range },
                  { 'kernel':['rbf'], 'C':C_range, 'gamma':gamma_ramge}]
    
    clf = GridSearchCV(svr, parameters, cv=CV_FOLDS, n_jobs = parallel, verbose=True)
    
    clf.fit(X_tr,y_tr)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    if print_CV_report:
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        print("mean \tstd \t\tparams")
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
    
    print()
    
    print("Detailed classification report on test data:")
    y_true, y_pred = y_te, clf.predict(X_te)
    print(classification_report(y_true, y_pred))
    print()
        

if __name__ == '__main__':
    X_tr,y_tr,X_te,y_te = loaddata("output-feature-engineering.csv")
    report_svm(X_tr,y_tr,X_te,y_te, True)