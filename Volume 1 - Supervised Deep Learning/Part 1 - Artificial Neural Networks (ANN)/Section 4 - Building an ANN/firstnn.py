#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:48:09 2018

@author: shivam
""

import numpy as np
import matplotlib as plt
import pandas as pd

df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:13].values
Y = df.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Encoder =  LabelEncoder()
X[:, 1] = Encoder.fit(X[:, 1]).transform(X[:, 1])
X[:, 2] = Encoder.fit(X[:, 2]).transform(X[:, 2])
oneHotEncoder = OneHotEncoder(categorical_features = [1])
X = oneHotEncoder.fit(X).transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
X_train = Scaler.fit(X_train).transform(X_train)
X_test = Scaler.fit(X_test).transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

clf = Sequential()
clf.add(Dense(output_dim = 6, init = 'uniform',activation='relu',input_dim = 11))
clf.add(Dense(output_dim = 6, init = 'uniform',activation='relu'))
clf.add(Dense(units = 1, init = 'uniform',activation='sigmoid'))

clf.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics =['accuracy'])

clf.fit(X_train,Y_train,batch_size=10, nb_epoch =100)

Y_pred = clf.predict(X_test)
y_pred = (Y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_Selction import cross_val_score
def build_classifier():
  clf = Sequential()
  clf.add(Dense(output_dim = 6, init = 'uniform',activation='relu',input_dim = 11))
  clf.add(Dense(output_dim = 6, init = 'uniform',activation='relu'))
  clf.add(Dense(units = 1, init = 'uniform',activation='sigmoid'))
  clf.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics =['accuracy'])
  return clf

clf =KerasClassifier(build_fn= build_classifier,batch_size=10,nb_epoch =100)  









