from sklearn import svm
import cvxpy as cvx
import numpy as np
import math
import numpy.linalg as linalg
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
import math
import cvxopt
from sklearn.metrics import accuracy_score

sigma=1

def cal(a,b):
    s=0.0
    for i in range(len(a)):
        s=s+(a[i]-b[i])*(a[i]-b[i])
    s=s*-1;
    s/=2*(sigma*sigma);
    s=math.exp(s)
    return s

def dis(a,b):
    r=0
    for i in range(len(a)):
        r+=(a[i]-b[i])*(a[i]-b[i])
    return r


def one_class_svm(X_train,X_test,y_train,y_test,digit):
    print('For digit '+str(digit))
    clf = svm.OneClassSVM(kernel='rbf', gamma=0.0001, nu=0.01)
    clf.fit(X_train, y_train)
    print('Built in classifiers accuracy is ' + str(accuracy_score(y_train, clf.predict(X_train)) * 100) + '%' + str(accuracy_score(y_test, clf.predict(X_test)) * 100) + '%')
    #print()
    return


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
    end=gc-1
    rang.append([start,end])


def conv_data_apply_pca(X1):
    optk = 0
    for k in range(1, len(X1[0])):
        pca = PCA(k);
        pca.fit(X1);
        if np.sum(pca.explained_variance_ratio_) > 0.95:
            print('opt is ', k)
            optk = k;
            break
    return [pca.transform(X1), pca.components_, optk]

X=np.array(X)
print(rang)



for i in range(10):
    yi=[-1 for i in range(len(y))]
    yi=np.array(yi);
    yi[rang[i][0]:rang[i][1]+1]=1
    X_train=X[rang[i][0]:rang[i][1]+1]
    X_test=X
    y_train=[1.0 for i in range(rang[i][1]-rang[i][0]+1)]
    y_test=yi
    [X_train,eig,optk]=conv_data_apply_pca(X_train)
    X_test1=np.dot(eig,X_test.T).T
    one_class_svm(X_train,X_test1,y_train,y_test,i)


