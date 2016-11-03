from data_utils import loaddata
from svm_utils import report_svm

#Run with all features
X_tr,y_tr,X_te,y_te = loaddata("output-feature-engineering.csv")
report_svm(X_tr,y_tr,X_te,y_te, True) 