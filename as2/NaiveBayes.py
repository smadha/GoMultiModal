from sklearn import datasets
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#y_pred = gnb.fit(training_X, training_Y).predict(X_te)
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))

from data_utils import loaddata
import numpy as np
X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",0)
training_X = [val for sublist in X_tr for val in sublist]
training_Y = [val for sublist in y_tr for val in sublist]
y_pred = gnb.fit(training_X, training_Y).predict(X_te)

print("Accuracy for Exp 1: %f" % (float((y_te != y_pred).sum())/float(X_te.shape[0])))
print "Confusion matrix", confusion_matrix(y_te, y_pred)
print(classification_report(y_te, y_pred))


X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",1)
training_X = [val for sublist in X_tr for val in sublist]
training_Y = [val for sublist in y_tr for val in sublist]
y_pred = gnb.fit(training_X, training_Y).predict(X_te)

print("Accuracy for Exp 2: %f" % (float((y_te != y_pred).sum())/float(X_te.shape[0])))
print "Confusion matrix", confusion_matrix(y_te, y_pred)
print(classification_report(y_te, y_pred))

X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",2)
training_X = [val for sublist in X_tr for val in sublist]
training_Y = [val for sublist in y_tr for val in sublist]
y_pred = gnb.fit(training_X, training_Y).predict(X_te)

print("Accuracy for Exp 3: %f" % (float((y_te != y_pred).sum())/float(X_te.shape[0])))
print "Confusion matrix", confusion_matrix(y_te, y_pred)
print(classification_report(y_te, y_pred))

X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-multimodal.csv",3)
training_X = [val for sublist in X_tr for val in sublist]
training_Y = [val for sublist in y_tr for val in sublist]
y_pred = gnb.fit(training_X, training_Y).predict(X_te)

print("Accuracy for Exp 4: %f" % (float((y_te != y_pred).sum())/float(X_te.shape[0])))
print "Confusion matrix", confusion_matrix(y_te, y_pred)
print(classification_report(y_te, y_pred))
