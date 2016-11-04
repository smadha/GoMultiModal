from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_utils import loaddata, get_custom_validation_sets

C_range = [ 10 ** i for i in range(-3, 3) ]

print (C_range)

# Number of parallel jobs
parallel = 1


def report_svm(X_tr, y_tr, X_te, y_te, print_CV_report=False):
    print "Running linear SVM"
    cv_custom, cv_custom_y = get_custom_validation_sets(X_tr, y_tr)
    
    # a array of ["C value","training acc","validation acc"] for each validation set
    results = [[]]*len(cv_custom)
    
    for C in C_range:
        clf = SVC(kernel='linear', C=C)
        for idx, cv_set in enumerate(cv_custom):
            
            training = cv_set[0]
            testing = cv_set[1]
            
            training_y = cv_custom_y[idx][0]
            testing_y = cv_custom_y[idx][1]
            
            clf.fit(training,training_y)
            
            y_true, y_pred =testing_y, clf.predict(testing)
            valid_score = accuracy_score(y_true, y_pred)
            
            y_true, y_pred =training_y, clf.predict(training)
            training_score = accuracy_score(y_true, y_pred)
                
            # if validation accuracy is higher for this C update results
            if results[idx] == [] or valid_score > results[idx][2]:
                results[idx] = [C,training_score,valid_score]
    
    
    print "\t".join(["C value","training acc","validation acc","Testing acc"])
           
    for idx, res in enumerate(results):
        clf = SVC(kernel='linear', C=res[0])
        
        training = cv_custom[idx][0]
        validation = cv_custom[idx][1]
        
        training_y = cv_custom_y[idx][0]
        validation_y = cv_custom_y[idx][1]
        
        clf.fit(training + validation, training_y + validation_y)
        
        y_true, y_pred = y_te, clf.predict(X_te)
        test_score = accuracy_score(y_true, y_pred)
        
        print "\t".join(map(str, res)) + "\t" + str(test_score)
        
        

if __name__ == '__main__':
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv")
    report_svm(X_tr, y_tr, X_te, y_te, True)
