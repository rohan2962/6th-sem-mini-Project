from sklearn import svm
import cvxpy as cvx
import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd
import math
import cvxopt
from sklearn.metrics import accuracy_score

X=[]
y=[]
f=[]
for i in range(10):
    #X.append([])
    #y.append([])
    f.append(open('features_'+str(i)+'.txt','r'))


for j in range(10):
    fp=f[j]
    for line in fp:
        #print(line)
        arr=[]
        s=""
        for i in range(len(line)):
            if line[i]==',':
                arr.append(int(s))
                s=""
            else:
                s=s+line[i]
        #print(type(s),s)
        arr.append(int(s))
        X.append(arr[0:200])
        y.append(arr[200])

clf=svm.SVC(kernel='rbf',gamma=0.0001)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)
print(y_test[0:10])
clf.fit(X_train,y_train)
print(clf.predict(X_test)[0:10])
print(accuracy_score(y_test,clf.predict(X_test)))




