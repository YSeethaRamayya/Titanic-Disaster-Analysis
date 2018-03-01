# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:35:10 2017

@author: ramayya
"""
#%reset -f


import pandas as pd

dataset=pd.read_csv("train.csv")
X=dataset.iloc[:,[0,2,4]].values
y=dataset.iloc[:,1].values

#Preprocessing Categorical Data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder=LabelEncoder()
X[:,2]=labelencoder.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[2])
onehotencoder.fit_transform(X).toarray()

#Splitting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=0)

#Fitting regression on training_set
from sklearn.svm import SVC
classifier=SVC(kernel="linear", random_state=0)
classifier.fit(X_train,y_train)

#Predicting test_set results
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

