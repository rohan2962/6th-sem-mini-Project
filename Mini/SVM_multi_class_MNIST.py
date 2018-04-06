from sklearn import svm
import cvxpy as cvx
import math
import numpy.linalg as linalg
import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd
import cvxopt
from sklearn.metrics import accuracy_score


X=[]
y=[]
f=[]

for i in range(10):
    #X.append([])
    #y.append([])
    f.append(open('features_mnist_more_'+str(i)+'.txt','r'))

#print(f)
rang=[]
gc=0
X2d=[[]for i in range(10)]
for it in range(10):
    fp=f[it]
    start=-1
    end=-1
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
        if start==-1:
            #print(start)
            start=gc
        arr.append(int(s))
        gc=gc+1
        X.append(arr[0:274])
        y.append(arr[274])
        #print(arr[1000])
    end=gc-1
    rang.append([start,end])

print(X[0])
print(y[0:40])

X=np.array(X)
print(rang)

mean=np.mean(X,axis=0)
normalized=X-mean
cov_matrix=np.cov(normalized.T)
eigen_values,eigen_vectors=linalg.eig(cov_matrix)
#eigen_vectors=eigen_vectors[:,0:20]
#conv_data=np.dot(eigen_vectors.T,normalized.T).T
#X=conv_data
#print(X[0])

optk=20
'''for k in range(1,len(X[0])+1):
    eigen_vectors_1=eigen_vectors[:,0:k]
    conv_data=np.dot(eigen_vectors_1.T,normalized.T).T
    X_approx=np.dot(eigen_vectors_1,conv_data.T).T
    #print(np.array(X_approx).shape)
    #print(np.array(normalized).shape)
    #print(len(normalized))
    a=0
    b=0
    for i in range(len(normalized)):
        a=a+np.sum(np.array(normalized[i]-X_approx[i])*np.array(normalized[i]-X_approx[i]))
        b=b+np.sum(np.array(normalized[i])*np.array(normalized[i]))
    print(k, (a / b).real)
    if (a / b).real <= 0.01:
        optk = k
        break
'''

#print(optk)
eigen_vectors=eigen_vectors[:,0:optk]
print(eigen_vectors)
print()
print()
print(eigen_values[0:20])
conv_data=np.dot(eigen_vectors.T,normalized.T).T
X=conv_data.real


X_train=[]
X_test=[]
y_test=[]
y_train=[]
X2d=[[]for i in range(10)]
for i in range(len(X)):
    X2d[y[i]].append(X[i])
print(len(X2d[0]))
print(len(X2d[0][0]))
for i in range(10):
    for j in range(4000):
        y_train.append(i)
        X_train.append(X2d[i][j])
    for j in range(4000,len(X2d[i])):
        y_test.append(i)
        X_test.append(X2d[i][j])
bsvm=svm.SVC(gamma=0.00005,C=0.5)
print(bsvm)
bsvm.fit(X_train,y_train)
print(accuracy_score(bsvm.predict(X_test),y_test))


