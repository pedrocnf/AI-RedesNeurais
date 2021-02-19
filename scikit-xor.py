# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:41:58 2020

@author: pedro
"""
import numpy as np
from sklearn.linear_model import SGDClassifier
X = np.array([[0,0], [0,1]])
y = np.array([[0], [1]])
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)
X_new = np.array([[1,0], [1,1]])
y_new = np.array([[1], [0]])
clf.partial_fit(X_new,y_new)