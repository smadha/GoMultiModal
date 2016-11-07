'''
Feed forward neural network with reasonable values of regularization coef and 3 hidden layers

Change  ~/.keras/keras.json to default to theano or install tensolflow

Theano should come automatically with Keras. See http://deeplearning.net/software/theano/install.html if you face issues
We don't need fully optimised Theano installation as data is less 
'''


from keras.layers import Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import numpy as np

from data_utils import loaddata, get_custom_validation_sets

def genmodel(num_units, actfn='relu', reg_coeff=0.0, last_act='softmax'):
    ''' Generate a neural network model of approporiate architecture
    Args:
        num_units: architecture of network in the format [n1, n2, ... , nL]
        actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
        reg_coeff: L2-regularization coefficient
        last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
    Output:
        model: Keras sequential model with appropriate fully-connected architecture
    '''

    model = Sequential()
    for i in range(1, len(num_units)):
        if i == 1 and i < len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=actfn,
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == 1 and i == len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=last_act,
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i < len(num_units) - 1:
            model.add(Dense(output_dim=num_units[i], activation=actfn,
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == len(num_units) - 1:
            model.add(Dense(output_dim=num_units[i], activation=last_act,
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
    return model

def transform_label(labels):
    '''
    Returns list of labels as list of [0/1 , 1/0, , 1/0]] 
    if label = 1 [0, 0, 1]
    if label = 0 [0, 1, 0]
    if label = -1 [1, 0, 0]
    '''
    labels_new = []
    for label in labels:
        label_new = [0.0, 0.0, 0.0]
        label_new[int(label) + 1] = 1.0
        labels_new.append(label_new)
    
    return np.array(labels_new)


def run_nn(X_tr, y_tr, X_te, y_te, validation_tuple):
    X_tr, y_tr, X_te, y_te = np.array(X_tr), np.array(y_tr), np.array(X_te), np.array(y_te)
    validation_tuple = (np.array(validation_tuple[0]) , np.array(validation_tuple[1]))
    
    # saving a copy of multiclass notation for computing accuracy
    y_tr_multiclass = y_tr
    y_valid_multiclass = validation_tuple[1]
    
    y_tr = transform_label(y_tr)
    validation_tuple = (validation_tuple[0], transform_label(validation_tuple[1]))
    
    print X_tr.shape
    print y_tr.shape
    print validation_tuple[0].shape, validation_tuple[1].shape
    
    reg_coeff = 5e-06
    momentum = 0.99
    eStop = True
    sgd_Nesterov = True
    sgd_lr = 5e-2
    sgd_decay = 5e-05
    arch = [len(X_tr[0]),100,50,len(y_tr[0])]
    batch_size=5
    nb_epoch=5
    verbose = False
    
    call_ES = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')
    
    # Generate Model
    model = genmodel(num_units=arch, reg_coeff=reg_coeff)
    # Compile Model
    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=momentum,
        nesterov=sgd_Nesterov)
    
    model.compile(loss='MSE', optimizer=sgd,
        metrics=['accuracy'])
    # Train Model
    if eStop:
        model.fit(X_tr, y_tr, nb_epoch=nb_epoch, batch_size=batch_size,
        verbose=verbose, callbacks=[call_ES], validation_split=None,
        validation_data=validation_tuple, shuffle=False)
    else:
        model.fit(X_tr, y_tr, nb_epoch=nb_epoch, batch_size=batch_size,
            verbose=verbose)
    
    # Save model
    model.save("model_deep.h5")
    print("Saved model to disk")
    
    print ""
    print "<REPORT>"
    y_true, y_pred = np.array(y_tr_multiclass) + 1, model.predict_classes(X_tr, verbose=verbose)
    print "Training accuracy", accuracy_score(y_true, y_pred)
    
    y_true, y_pred = np.array(y_valid_multiclass) + 1, model.predict_classes(validation_tuple[0], verbose=verbose)
    print "Validation accuracy", accuracy_score(y_true, y_pred)

    y_true, y_pred = np.array(y_te) + 1, model.predict_classes(X_te, verbose=verbose)
    print "Testing accuracy", accuracy_score(y_true, y_pred)
    
    print "Confusion matrix", confusion_matrix(y_true, y_pred) 
    print y_true
    print y_pred
    print "Detailed classification report on test data:"
    
    print(classification_report(y_true, y_pred))
    print "</REPORT>"
    print ""


# Run with all features
X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-acoustic.csv",0,3,True)

cv_custom, cv_custom_y = get_custom_validation_sets(X_tr, y_tr)


for idx, cv_set in enumerate(cv_custom):
            
    training = cv_set[0]
    valid = cv_set[1]
    
    training_y = cv_custom_y[idx][0]
    valid_y = cv_custom_y[idx][1]
    
    run_nn(training, training_y, X_te, y_te, (valid, valid_y))
    break

X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-acoustic.csv",1,3,True)

cv_custom, cv_custom_y = get_custom_validation_sets(X_tr, y_tr)


for idx, cv_set in enumerate(cv_custom):

    training = cv_set[0]
    valid = cv_set[1]

    training_y = cv_custom_y[idx][0]
    valid_y = cv_custom_y[idx][1]

    run_nn(training, training_y, X_te, y_te, (valid, valid_y))
    break

X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-acoustic.csv",2,3,True)

cv_custom, cv_custom_y = get_custom_validation_sets(X_tr, y_tr)


for idx, cv_set in enumerate(cv_custom):

    training = cv_set[0]
    valid = cv_set[1]

    training_y = cv_custom_y[idx][0]
    valid_y = cv_custom_y[idx][1]

    run_nn(training, training_y, X_te, y_te, (valid, valid_y))
    break

X_tr, y_tr, X_te, y_te = loaddata("output-feature-engineering-acoustic.csv",3,3,True)

cv_custom, cv_custom_y = get_custom_validation_sets(X_tr, y_tr)


for idx, cv_set in enumerate(cv_custom):

    training = cv_set[0]
    valid = cv_set[1]

    training_y = cv_custom_y[idx][0]
    valid_y = cv_custom_y[idx][1]

    run_nn(training, training_y, X_te, y_te, (valid, valid_y))
    break



