from sklearn import svm
import cvxpy as cvx
import numpy as np
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
import math
from sklearn.decomposition import PCA
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
    print('For digit ' + str(digit))
    clf = svm.OneClassSVM(kernel='rbf', gamma=0.01, nu=0.01)
    clf.fit(X_train, y_train)
    print('Built in classifiers accuracy is ' + str(accuracy_score(y_test, clf.predict(X_test)) * 100) + '%')
    return (accuracy_score(y_test, clf.predict(X_test)) * 100)

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

X=np.array(X)
print(rang)


gk=[]
fraction_retain=[]
def conv_data_apply_pca(X1):
    optk=0
    flag=0
    print(len(X1[0]))
    for k in range(1,len(X1[0])):
        pca=PCA(k);
        pca.fit(X1);
        gk.append(k)
        fraction_retain.append(np.sum(pca.explained_variance_ratio_))
        if np.sum(pca.explained_variance_ratio_)>0.95 and flag==0:
            print('opt is ',k)
            optk=k;
            flag=1
    pca = PCA(optk);
    pca.fit(X1);
    return  [pca.transform(X1),pca.components_,optk]


def normal_pca(X,k):
    pca = PCA(k);
    pca.fit(X);
    return [pca.transform(X), pca.components_]

dig=[i for i in range(10)]
for i in range(1):
    new_x = X[rang[i][0]:rang[i][1] + 1]
    no_of_fe=[]
    acc=[]
    [new1_x, eigen_vector, optk] = conv_data_apply_pca(new_x)
    plt.plot(gk,fraction_retain)
    plt.plot([optk], [fraction_retain[optk-1]],marker='o',color='r')
    plt.xlabel('No of principal components')
    plt.ylabel('Fraction of variance retained')
    #plt.show()
    plt.savefig('PCA with fraction of variance retained.png')
    save_acc=0
    #plt.savefig('Fraction of retained variance vs no of principal components for digit '+str(i))
    for j in range(1,gk[len(gk)-1],20):
        print(j,len(gk))
        [new1_x, eigen_vector] = normal_pca(new_x,j)
        X1 = np.dot(X, eigen_vector.T)
        X_test = X1
        yi=[-1 for k in range(len(y))]
        yi=np.array(yi);
        yi[rang[i][0]:rang[i][1]+1]=1
        X_train=X1[rang[i][0]:rang[i][1]+1-2000]
        y_train=[1.0 for k in range(rang[i][1]-rang[i][0]+1-2000)]
        y_test=yi
        ac1=one_class_svm(X_train,X_test,y_train,y_test,i)
        no_of_fe.append(j)
        acc.append(ac1)

    [new1_x, eigen_vector] = normal_pca(new_x,optk)
    X1 = np.dot(X, eigen_vector.T)
    X_test = X1
    yi = [-1 for k in range(len(y))]
    yi = np.array(yi);
    yi[rang[i][0]:rang[i][1] + 1] = 1
    X_train = X1[rang[i][0]:rang[i][1] + 1 - 2000]
    y_train = [1.0 for k in range(rang[i][1] - rang[i][0] + 1 - 2000)]
    y_test = yi
    ac1 = one_class_svm(X_train, X_test, y_train, y_test, i)
    no_of_fe.append(optk)
    acc.append(ac1)
    save_acc=ac1

    plt.clf()
    plt.plot(no_of_fe,acc)
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    #plt.show()
    plt.savefig('Acc with PCA.png')


