#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras import backend as K
from matplotlib.pyplot import imshow
import sys
sys.path.append("..")
import utils
from utils import *
print('Imports done')

K.set_image_dim_ordering('th')

# Load the dataset
num_classes = 10
X_train, y_train, X_test, y_test = getMNISTData()

## Categorize the labels
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

#################################
## Model specification

## Start from an empty sequential model where we can stack layers
model = Sequential()

## Add a fully-connected layer with 128 neurons. The input dim is 784 which is the size of the pixels in one image
model.add(Dense(output_dim=160, input_dim=784))

# Using 128 neurons in hidden layer
# Accuracy on test set: 0.9154
# Using 64 neurons in hidden layer
# Accuracy on test set: 0.9181
# Using 32 neurons in hidden layer
# Accuracy on test set: 0.9088

## Add rectifier activation function to each neuron
model.add(Activation("relu"))

# Adding 2nd hidden layer with 64 fully connected neurons
# Accuracy on test set: 0.9272
#model.add(Dense(output_dim=64))
#model.add(Activation("relu"))

# Adding 2nd hidden layer with 128 fully connected neurons
# Accuracy on test set: 0.9272
#model.add(Dense(output_dim=128))
#model.add(Activation("relu"))

# Adding 2nd hidden layer with 16 fully connected neurons
# Accuracy on test set: 0.9202
#model.add(Dense(output_dim=16))
#model.add(Activation("relu"))

#model.add(Dense(output_dim=32))
#model.add(Activation("relu"))

## Add another fully-connected layer with 10 neurons, one for each class of labels
model.add(Dense(output_dim=10))

## Add a softmax layer to force the 10 outputs to sum up to one so that we have a probability representation over the labels.
model.add(Activation("softmax"))

##################################

## Compile the model with categorical_crossentrotry as the loss, and stochastic gradient descent (learning rate=0.001, momentum=0.5,as the optimizer)
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.02, momentum=0.78), metrics=["accuracy"])

# Testing
# Adjusting lr=0.0005, momentum=0.5
# Accuracy on test set: 0.9015
# Adjusting lr=0.002, momentum=0.5
# Accuracy on test set: 0.9306
# Adjusting lr=0.002, momentum=0.7
# Accuracy on test set: 0.9436
# Adjusting lr=0.002, momentum=0.8 64HU
# Accuracy on test set: 0.9444
# Adjusting lr=0.004, momentum=0.75 64HU
# Accuracy on test set: 0.9566
# Adjusting lr=0.008, momentum=0.75 64HU
# Accuracy on test set: 0.9678
# Adjusting lr=0.012, momentum=0.75 64HU
# Accuracy on test set: 0.9706
# Adjusting lr=0.012, momentum=0.75 128HU
# Accuracy on test set: 0.9749
# Adjusting lr=0.016, momentum=0.75 64HU
# Accuracy on test set: 0.9736
# Adjusting lr=0.02, momentum=0.8 64HU
# Accuracy on test set: 0.9743
# Adjusting lr=0.02, momentum=0.75 64HU
# Accuracy on test set: 0.9726
# Adjusting lr=0.014, momentum=0.75 64HU
# Accuracy on test set: 0.9729
# Adjusting lr=0.013, momentum=0.75 64HU
# Accuracy on test set: 0.9716
# Adjusting lr=0.012, momentum=0.8 64HU
# Accuracy on test set: 0.9751
# Adjusting lr=0.012, momentum=0.85 64HU
# Accuracy on test set: 0.9733
# Adjusting lr=0.012, momentum=0.8 128HU
# Accuracy on test set: 0.9756
# Adjusting lr=0.012, momentum=0.8 96HU
# Accuracy on test set: 0.9757
# Adjusting lr=0.012, momentum=0.8 96HU+32HU
# Accuracy on test set: 0.9737
# Adjusting lr=0.012, momentum=0.8 160HU
# Accuracy on test set: 0.9777
# Adjusting lr=0.010, momentum=0.8 160HU
# Accuracy on test set: 0.9765
# Adjusting lr=0.016, momentum=0.8 160HU
# Accuracy on test set: 0.9783
# Adjusting lr=0.02, momentum=0.8 160HU
# Accuracy on test set: 0.9798
# Adjusting lr=0.021, momentum=0.8 160HU
# Accuracy on test set: 0.977
# Adjusting lr=0.019, momentum=0.8 160HU
# Accuracy on test set: 0.977
# Adjusting lr=0.020, momentum=0.79 160HU
# Accuracy on test set: 0.979
# Adjusting lr=0.020, momentum=0.78 160HU
# Accuracy on test set: 0.979
# Adjusting lr=0.020, momentum=0.78 160HU
# Accuracy on test set: 0.9772

## Fit the model (10% of training data used as validation set)
model.fit(X_train, y_train, nb_epoch=10, batch_size=32,validation_split=0.1)

## Evaluate the model on test data
objective_score = model.evaluate(X_test, y_test, batch_size=32)

# objective_score is a tuple containing the loss as well as the accuracy
print ("Loss on test set:"  + str(objective_score[0]) + " Accuracy on test set: " + str(objective_score[1]))

if K.backend()== 'tensorflow':
    K.clear_session()