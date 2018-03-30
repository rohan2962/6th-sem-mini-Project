from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.cross_validation import train_test_split
import random
import math
import copy

def similarity(l,j,sigma):
    val=0
    for i in range(len(l)):
        val=val+(l[i]-j[i])*(l[i]-j[i])
    val=val/(2*sigma*sigma)
    val=math.exp(-1*val)
    return val

def dot(a,b):
    ans=0
   # print(len(a),len(b))
    for i in range(len(a)):
        ans+=a[i]*b[i]
    return ans

def svm_first(X,y,n_epochs,sigma,C,learning_rate):
    theta=[random.random() for i in range(len(X))]
    #print(theta)
    for i in range(1):
        for j in range(len(X)):
            print(j)
            f=[]
            for l in X:
                val=similarity(l,X[j],sigma)
                f.append(val)
            #cost=cost+C*(y[j]*math.log(1.0/(1.0+math.exp(-1*dot(theta,f))),math.e)+(1-y[j])*math.log(1.0-(1.0/(1.0+math.exp(-1*dot(theta,f)))),math.e))
            new_theta=copy.deepcopy(theta)
            e_value=dot(theta,f)
            for l in range(len(theta)):
                x=0
                x=C*((1-y[j])*(2.718*e_value)+y[j]*(1-2.718*e_value))
                new_theta[l]=new_theta[l]-learning_rate*x
            theta=copy.deepcopy(new_theta)
    return theta

def predict(theta,X,x,sigma):
    ans=0
    f = []
    for l in X:
        val = similarity(l, x, sigma)
        f.append(val)
    ans=dot(theta,f)
    if ans>=0:
        return 1
    else:
        return 0

if __name__ == "__main__":
    print('Hello')
    f0 = open('features_0.txt', 'r')
    f6 = open('features_6.txt', 'r')
    f8 = open('features_8.txt', 'r')
    f9 = open('features_9.txt', 'r')
    X0, y0, X6, y6, X8, y8, X9, y9 = [], [], [], [], [], [], [], []
    for pp in [0, 6, 8, 9]:
        X, y = [], []
        fp = 0
        if pp == 0:
            fp = f0
        elif pp == 6:
            fp = f6
        elif pp == 8:
            fp = f8
        elif pp == 9:
            fp = f9
        for line in fp:
            # print(line)
            arr = []
            s = ""
            for i in range(len(line)):
                if line[i] == ',':
                    arr.append(int(s))
                    s = ""
                else:
                    s = s + line[i]
            arr.append(int(s))
            X.append(arr[0:200])
            y.append(arr[200])
        if pp == 0:
            X0, y0 = X, y
        elif pp == 6:
            X6, y6 = X, y
        elif pp == 8:
            X8, y8 = X, y
        elif pp == 9:
            X9, y9 = X, y
            # print(y)
    clf = svm.SVC()
    X = []

    for i in X0:
        X.append(i)
    for i in X6:
        X.append(i)
    for i in X8:
        X.append(i)
    for i in X9:
        X.append(i)
    X = np.array(X)
    print(X.shape)
    y = []
    print(np.array(X0).shape, np.array(y0).shape)
    for i in y0:
        y.append(1)
    for i in y6:
        y.append(0)
    for i in y8:
        y.append(0)
    for i in y9:
        y.append(0)
    y = np.array(y)
    y.reshape([-1, ])
    print(y.shape)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    theta=svm_first(X_train,y_train,100,12,1,0.001)
    acc=0.0
    for i in range(len(X_test)):
        if predict(theta,X_train,X_test[i],1)==y_test[i]:
            acc+=1
    print(acc/len(y_test))
    clf=svm.SVC()
    print(clf)
    clf.fit(X_train,y_train)
    print(accuracy_score(y_test,clf.predict(X_test)))