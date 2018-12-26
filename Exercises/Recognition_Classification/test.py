import os
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

X =  list(range(10))
print (X)
y = [x*x for x in X]
print (y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,test_size=0.25, random_state=0)

#X_train, y_train = X[-3:], y[-3:] 
#X_test, y_test = X[:-3], y[:-3]

print ("X_train: ", X_train)
print ("y_train: ", y_train)
print("X_test: ", X_test)
print ("y_test: ", y_test)

X_train, y_train = shuffle(X_train, y_train,n_samples = 3)
print ("X_train: ", X_train)
print ("y_train: ", y_train)
