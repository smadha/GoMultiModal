from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import csv
import numpy as np

from data_utils import loaddata, get_custom_validation_sets
from as1.plot import plot_histogram, plot_box, plot_scatter, plot_xy_histogram

C_range = [ 10 ** i for i in range(-2, 3) ]
gamma_range = [10 ** i for i in range(-2, 3)]
print (C_range)

# Number of parallel jobs
parallel = 1

def train_evaluate(cv_custom, cv_custom_y, clf, valid_fold):
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
    training = cv_custom[idx][0]
    validation = cv_custom[idx][1]

    training_y = cv_custom_y[idx][0]
    validation_y = cv_custom_y[idx][1]
    clf.fit(training + validation, training_y + validation_y)
    y_true, y_pred = y_te, clf.predict(X_te)
    test_score = accuracy_score(y_true, y_pred)

    result = [sum_train/valid_fold,sum_valid/valid_fold,test_score,abs(sum_valid/valid_fold - test_score) ]
    return result,y_pred,sum_valid/valid_fold

def report_svm(X_tr, y_tr, X_te, y_te, print_CV_report=False, kernel_value='linear'):
    print "Running linear SVM"
    cv_custom, cv_custom_y = get_custom_validation_sets(X_tr, y_tr)
    
    # a array of ["C value","training acc","validation acc"] for each validation set
    results = []
    valid_fold = len(cv_custom)
    prev = 0
    to_return = []
    for id_C,C in enumerate(C_range):
        if(kernel_value!='linear'):
            for id_g,gamma in enumerate(gamma_range):
                clf = SVC(C=C, gamma=gamma)
                result, y_pred, validation_accuracy = train_evaluate(cv_custom, cv_custom_y, clf, valid_fold)
                result.insert(0,'C = '+str(C)+' gamma = '+str(gamma))
                results.append(result)
                if(validation_accuracy>prev):
                    prev = validation_accuracy
                    to_return = y_pred
        else:
            clf = SVC(kernel='linear', C=C)
            result, y_pred, validation_accuracy = train_evaluate(cv_custom, cv_custom_y, clf, valid_fold)
            result.insert(0,'C = '+str(C))
            results.append(result)
            if(validation_accuracy>prev):
                prev = validation_accuracy
                to_return = y_pred


    print "\t".join(["Hyper parameter","training acc","validation acc","Testing acc","Difference"])
           
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
    return to_return
        

if __name__ == '__main__':
    print ("4 fold multi modal linear")
    result = []
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0)
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1)
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2)
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3)
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))

    '''print ("4 fold multi modal with selected features")
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0 , 3, False, [0, 1, 2, 5, 6, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1, 3, False, [0, 1, 2, 5, 6, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2, 3, False, [0, 1, 2, 5, 6, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3, 3, False, [0, 1, 2, 5, 6, 11, 13, 16, 18])
    report_svm(X_tr, y_tr, X_te, y_te, True)
    print ("4 fold multi modal without speaker independence")
    #X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0, 3, True)
    #report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1, 3, True)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2, 3, True)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    #X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3, 3, True)
    #report_svm(X_tr, y_tr, X_te, y_te, True)

    print ("3 fold multi modal without speaker independence")
    #X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0, 2, True,range(0,19,1),1.0/3.0)
    #report_svm(X_tr, y_tr, X_te, y_te, True)
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1, 2, True,range(0,19,1),1.0/3.0)
    report_svm(X_tr, y_tr, X_te, y_te, True)
    #X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2, 2, True,range(0,19,1),1.0/3.0)
    report_svm(X_tr, y_tr, X_te, y_te, True)'''

    print ("4 fold acoustic modal")
    Acoustic = []
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0, 3, False, [0, 11, 13, 16, 18])
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1, 3, False, [0, 11, 13, 16, 18])
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2, 3, False, [0, 11, 13, 16, 18])
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3, 3, False, [0, 11, 13, 16, 18])
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))

    print ("4 fold visual modal")
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0, 3, False, [0, 1, 2, 5, 6, 18])
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1, 3, False, [0, 1, 2, 5, 6, 18])
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2, 3, False, [0, 1, 2, 5, 6, 18])
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3, 3, False, [0, 1, 2, 5, 6, 18])
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True))

    '''data_to_plot = [result[0],result[4]]
    plot_box(data_to_plot, "images", "Multimodal_acoustic_1",["Multimodal","Acoustic"])
    data_to_plot = [result[1],result[5]]
    plot_box(data_to_plot, "images", "Multimodal_acoustic_2",["Multimodal","Acoustic"])
    data_to_plot = [result[2],result[6]]
    plot_box(data_to_plot, "images", "Multimodal_acoustic_3",["Multimodal","Acoustic"])
    data_to_plot = [result[3],result[7]]
    plot_box(data_to_plot, "images", "Multimodal_acoustic_4",["Multimodal","Acoustic"])

    data_to_plot = [result[0],result[8]]
    plot_box(data_to_plot, "images", "Multimodal_visual_1",["Multimodal","Visual"])
    data_to_plot = [result[1],result[9]]
    plot_box(data_to_plot, "images", "Multimodal_visual_2",["Multimodal","Visual"])
    data_to_plot = [result[2],result[10]]
    plot_box(data_to_plot, "images", "Multimodal_visual_3",["Multimodal","Visual"])
    data_to_plot = [result[3],result[11]]
    plot_box(data_to_plot, "images", "Multimodal_visual_4",["Multimodal","Visual"])

    print ("4 fold multi modal RBF")
    result = []
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0)
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True,'RBF'))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1)
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True,'RBF'))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2)
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True,'RBF'))
    X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3)
    result.append(report_svm(X_tr, y_tr, X_te, y_te, True,'RBF'))

    result = list(map(list, zip(*result)))
    results = [[]]+result
    results[0] = ['Multimodal 1','Multimodal 2','Multimodal 3','Multimodal 4','Acoustic 1','Acoustic 2','Acoustic 3','Acoustic 4','Visual 1','Visual 2','Visual 3','Visual 4']
    with open("output-multimodal_diff_v3.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(results)'''

