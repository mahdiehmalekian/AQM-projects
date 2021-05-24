# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:20:58 2020

@author: Mahdieh Malekian
With help from many resources online!
"""

import numpy as np
import cvxopt
from cvxopt import matrix

# Linear kernel
def lin_ker (x, xp):
    return x @ xp

# Polynomial kernel
def poly_ker (x, xp, deg, r):
    #print ('x.shape, xp.shape', x.shape, xp.shape)
    return (r + x @ xp.T) ** deg

# Radial kernel
def radial_ker (x, xp, gamma):
    return np.exp(-gamma * np.linalg.norm (x - xp)**2)

# To find Lagrangian dual optimal solution:
def lag_dual_opt (X, y, K, C):
    N, p = X.shape
    y = y.astype('float')

    # To represent Lagrangian dual problem in a form solvable by cvxopt
    P = matrix(K * np.outer(y,y))
    q = matrix(-1.0 * np.ones(N))
    A = matrix(y, (1,N))
    b = matrix(0.0)

    if C is None:
        G = matrix(-1 * np.eye(N))
        h = matrix(np.zeros(N))
    else:
        G = matrix(np.vstack((-1 * np.eye(N), np.eye(N))))
        h = matrix(np.hstack((np.zeros(N), C * np.ones(N))))
        
    cvxopt.solvers.options['show_progress'] = False
    lag_dual_opt = cvxopt.solvers.qp(P, q, G, h, A, b)

    return np.array(lag_dual_opt['x']).reshape(-1)


class svm_AQM2020 (object):
    """The classes are assumed to be -1 and 1"""
    
    def __init__ (self, kernel='linear', C=None, gamma=1, degree=3, coef0=0.0):
        if kernel == 'linear':
            self.kernel = lin_ker
        elif kernel == 'poly':
            self.kernel = poly_ker
            self.deg, self.r = degree, coef0
        elif kernel == 'rbf':
            self.kernel = radial_ker
            self.gamma = gamma
            
        self.C = C
        if self.C is not None: self.C = float (self.C)
        
    def fit (self, X, y):
        N, p = X.shape
        y = y.astype('float')
        
        # The kernel matrix
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if self.kernel == lin_ker:
                    K[i,j] = self.kernel(X[i], X[j])
                elif self.kernel == poly_ker:
                    K[i,j] = self.kernel(X[i], X[j], self.deg, self.r)
                elif self.kernel == radial_ker:
                    K[i,j] = self.kernel(X[i], X[j], self.gamma)

        alpha = lag_dual_opt (X, y, K, self.C)
        
        # Only support vectors have nonzero alpha coefficients
        # Decide when to treat a number as zero
        accuracy = 1e-7
        sv_index = np.arange(N)[alpha > accuracy]
        
        # Only support vectors are significant:
        self.X = X[sv_index]
        self.y = y[sv_index]
        self.alpha = alpha[sv_index]
        
        # Find beta:
        self.beta = 0
        for i in range (len (self.alpha)):
            self.beta += self.alpha[i] * self.y[i] * self.X[i]

        # Find beta_0:
        self.beta_0 = 0
        count = 0
        for i in range(len(self.alpha)):
            if abs (self.alpha[i] - self.C) > accuracy:
            # Any such vector does the job, but we take average for numerical stability
                count += 1
                self.beta_0 += (1.0/self.y[i]) - np.sum(self.alpha * self.y * K[sv_index[i],sv_index])
        self.beta_0 /= count

    def predict(self, X):
        """"Assumes X is a 2D array with p columns"""

        if self.kernel == lin_ker:
            return np.sign(X @ self.beta + self.beta_0)
        
        y_hat = np.zeros(len(X))
        
        if self.kernel == poly_ker:
            for i in range(len(X)):
                for j in range (len(self.X)):
                    y_hat [i] += self.alpha [j] * self.y[j] * self.kernel (X[i], self.X[j], self.deg, self.r)
        elif self.kernel == radial_ker:
            for i in range(len(X)):
                for j in range (len(self.X)):
                    y_hat [i] += self.alpha [j] * self.y[j] * self.kernel (X[i], self.X[j], self.gamma)
                    
        y_hat += self.beta_0
        
        return np.sign(y_hat)
    
    

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#-----------------------Test the code on Breast Cancer Dataset-----------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Import Libraries
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from timeit import default_timer as timer



# -----------------------
# Import Data sets
DataSet = pd.read_csv('./Data/BreastCancer.csv')
DataSet.dropna()
# DataSet.describe()  # Diagnostic
x_columns = np.r_[1:6, 8:10]
# x_columns = np.r_[1:10]
X = DataSet.iloc[:, x_columns].values 
y = DataSet.iloc[:, len(DataSet.columns) - 1].values
y [y == 0] =-1
# -----------------------------
# Splitting data in training and test set.
test_size = 0.1  # Ratio of test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) 
    

def confusion_matrix (X_test, y_test, y_predict):
    cm = (np.zeros(4).reshape((2,2)))
    for i in range (len(X_test)):
        if y_predict [i] == -1:
            if y_test [i] == -1: cm [0, 0] += 1
            else: cm [0, 1] += 1
        else:
            if y_test [i] == -1: cm [1, 0] += 1
            else: cm [1,1] += 1
    return cm

    
# -----------------------------Testing our code using confusion matrix:
    
# linear kernel:
clf_AQM_l = svm_AQM2020 (kernel ='linear', C=10)
clf_AQM_l.fit(X_train, y_train)
print ('Confusion matrix for our svm with linear kernel:')
print (confusion_matrix (X_test, y_test, clf_AQM_l.predict (X_test)))

# radial kernel:
clf_AQM_r = svm_AQM2020 (kernel ='rbf', C=1, gamma=1)
clf_AQM_r.fit(X_train, y_train)
print ('Confusion matrix for our svm with radial kernel:')
print (confusion_matrix (X_test, y_test, clf_AQM_r.predict (X_test)))

# polynomial kernel:
clf_AQM_p = svm_AQM2020 (kernel ='poly', C=1, degree=3, coef0=2)
clf_AQM_p.fit(X_train, y_train)
print ('Confusion matrix for our svm with polynomial kernel:')
print (confusion_matrix (X_test, y_test, clf_AQM_p.predict (X_test)))

# -----------------------------Comparing our code with sklearn svm:

def compare_our_svm_with_sklearn (X_train, y_train, X_test, kernel, C=None, gamma=1, coef0=0.0, degree=3):
    start_time = timer()
    clf = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0, degree=degree)
    clf.fit(X_train, y_train)
    clf.predict (X_test)
    print ('Takes sklearn %3.3f to do the computation with' %(timer() - start_time), kernel, 'kernel')
    
    start_time = timer()
    clf_AQM = svm_AQM2020 (kernel =kernel, C=C, gamma=gamma, coef0=coef0, degree=degree)
    clf_AQM.fit(X_train, y_train)
    clf_AQM.predict (X_test)
    print ('Takes our code %3.3f to do the computation with' %(timer() - start_time), kernel, 'kernel')
    
compare_our_svm_with_sklearn (X_train, y_train, X_test, 'linear', C=10)
compare_our_svm_with_sklearn (X_train, y_train, X_test, 'rbf', C=1, gamma=1)
compare_our_svm_with_sklearn (X_train, y_train, X_test, 'poly', C=1, degree=3, coef0=2)


