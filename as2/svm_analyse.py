from data_utils import loaddata
from svm_utils import report_svm

#Run with all features and 4 validation set
X_tr,y_tr,X_te,y_te = loaddata("output-feature-engineering-multimodal.csv")
report_svm(X_tr,y_tr,X_te,y_te, True) 

#Run with visual features and 4 validation set
X_tr,y_tr,X_te,y_te = loaddata("output-feature-engineering-multimodal.csv",col_selected=[1,2,5,6])
report_svm(X_tr,y_tr,X_te,y_te, True) 

#Run with acoustic features and 4 validation set
X_tr,y_tr,X_te,y_te = loaddata("output-feature-engineering-multimodal.csv",col_selected=[11,13,16,17])
report_svm(X_tr,y_tr,X_te,y_te, True) 

#Run with all features and 3 shuffled validation set
X_tr,y_tr,X_te,y_te = loaddata("output-feature-engineering-multimodal.csv",split=3, shuffle=True)
report_svm(X_tr,y_tr,X_te,y_te, True) 

