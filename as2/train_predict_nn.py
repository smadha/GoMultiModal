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

from sklearn.metrics import classification_report
import numpy as np

from data_utils import loaddata

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
        label_new = [0.0,0.0, 0.0]
        label_new[int(label)+1]=1.0
        labels_new.append(label_new)
    
    return labels_new

#Run with all features
X_tr,y_tr,X_te,y_te = loaddata("output-feature-engineering.csv")

y_tr = transform_label(y_tr)

print len(X_tr),len(X_tr[0])
print len(y_tr),len(y_tr[0])

reg_coeff = 5e-06
momentum = 0.99
eStop = True
sgd_Nesterov = True
sgd_lr = 5e-2
sgd_decay = 5e-05
arch = [len(X_tr[0]),100,50,len(y_tr[0])]
batch_size=500
nb_epoch=50
verbose=True

call_ES = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')

# Generate Model
model = genmodel(num_units=arch, reg_coeff=reg_coeff )
# Compile Model
sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=momentum, 
    nesterov=sgd_Nesterov)

model.compile(loss='MSE', optimizer=sgd, 
    metrics=['accuracy'])
# Train Model
if eStop:
    model.fit(X_tr,y_tr, nb_epoch=nb_epoch, batch_size=batch_size, 
    verbose=verbose, callbacks=[call_ES], validation_split=0.1, 
    validation_data=None, shuffle=True)
else:
    model.fit(X_tr,y_tr, nb_epoch=nb_epoch, batch_size=batch_size, 
        verbose=verbose)

# Save model
model.save("model_deep.h5")
print("Saved model to disk")


y_true, y_pred = np.array(y_te)+1, model.predict_classes(X_te, verbose=1)


print("Detailed classification report on test data:")

print(classification_report(y_true, y_pred))


    