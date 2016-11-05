from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_utils import loaddata, get_custom_validation_sets

C_range = [ 10 ** i for i in range(-2, 3) ]

print (C_range)

# Number of parallel jobs
parallel = 1


def report_svm(X_tr, y_tr, X_te, y_te, print_CV_report=False):
    print "Running linear SVM"
    cv_custom, cv_custom_y = get_custom_validation_sets(X_tr, y_tr)
    
    # a array of ["C value","training acc","validation acc"] for each validation set
    results = [[]]*len(C_range)
    valid_fold = len(cv_custom)
    
    for id_C,C in enumerate(C_range):
        clf = SVC(kernel='linear', C=C)
        sum_train = 0
        sum_valid = 0
        sum_test = 0
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

            y_true, y_pred = y_te, clf.predict(X_te)
            test_score = accuracy_score(y_true,y_pred)

            sum_train = sum_train + training_score
            sum_valid = sum_valid + valid_score
            sum_test = sum_test + test_score
            # if validation accuracy is higher for this C update results
            #if results[idx] == [] or valid_score > results[idx][2]:
            #    results[idx] = [C,training_score,valid_score]
        results[id_C] = [C,sum_train/valid_fold,sum_valid/valid_fold,sum_test/valid_fold]
    
    print "\t".join(["C value","training acc","validation acc","Testing acc"])
           
    for idx, res in enumerate(results):
        #clf = SVC(kernel='linear', C=res[0])
        
        #training = cv_custom[idx][0]
        #validation = cv_custom[idx][1]
        
        #training_y = cv_custom_y[idx][0]
        #validation_y = cv_custom_y[idx][1]
        
        #clf.fit(training + validation, training_y + validation_y)
        
        #y_true, y_pred = y_te, clf.predict(X_te)
        #test_score = accuracy_score(y_true, y_pred)
        
        print "\t".join(map(str, res)) #+ "\t" + str(test_score)
        
        

if __name__ == '__main__':
    print ("4 fold multi modal")
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    print ("4 fold multi modal with selected features")
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0 , 3, False, [0, 1, 2, 5, 6, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1, 3, False, [0, 1, 2, 5, 6, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2, 3, False, [0, 1, 2, 5, 6, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3, 3, False, [0, 1, 2, 5, 6, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    print ("4 fold multi modal without speaker independence")
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0, 3, True)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1, 3, True)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2, 3, True)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3, 3, True)
    report_svm(X_tr, y_tr, X_te, y_te, True)

    print ("3 fold multi modal without speaker independence")
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0, 2, True)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1, 2, True)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2, 2, True)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3, 2, True)
    report_svm(X_tr, y_tr, X_te, y_te, True)

    print ("4 fold acoustic modal")
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0, 3, False, [0, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1, 3, False, [0, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2, 3, False, [0, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3, 3, False, [0, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)

    print ("4 fold visual modal")
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0, 3, False, [0, 1, 2, 5, 6, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1, 3, False, [0, 1, 2, 5, 6, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2, 3, False, [0, 1, 2, 5, 6, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3, 3, False, [0, 1, 2, 5, 6, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
