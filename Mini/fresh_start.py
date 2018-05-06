from sklearn import svm
import cvxpy as cvx
import numpy as np
import math
import numpy.linalg as linalg
from sklearn.cross_validation import train_test_split
import pandas as pd
import math
import cvxopt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

def one_class_svm(X_train,X_test,y_train,y_test,digit):
    print('For digit '+str(digit))
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    print(X_train.shape,y_train.shape)
    clf = svm.OneClassSVM(kernel='rbf')

    C1 = np.arange(0.001,0.1,0.003);
    C2 = np.arange(0.1,0.9,0.1);
    #0.1 0.9 0.1
    #0.001 0.1 0.003

    clf1 = GridSearchCV(estimator=clf, param_grid=dict(gamma=C1,nu=C2), scoring='accuracy')
    clf1.fit(X_train,y_train)
    print(clf1.best_score_)
    params = clf1.best_params_
    gamma = params['gamma']
    nu = params['nu']
    clf=svm.OneClassSVM(kernel='rbf',gamma=gamma,nu=nu)
    clf.fit(X_train)
    res_self = clf.predict(X_train)
    res = clf.predict(X_test)
    print('score for class ',digit, np.average(res_self == y_train))
    print('Built in classifiers accuracy is ' + str(accuracy_score(y_test, clf.predict(X_test)) * 100) + '%')
    #print()
    return

X_mnist=[]
X_nhrr=[]
y_mnist=[]
y_nhrr=[]
rang_mnist=[]
rang_nhrr=[]
def read_file(fp,x,type):
        global gc
        start = -1
        end = -1
        X1 = []
        y1 = []
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
            # print(type(s),s)
            if start == -1:
                # print(start)
                start = gc
            arr.append(int(s))
            gc = gc + 1
            if type==1:
                X_mnist.append(arr[0:x])
                y_mnist.append(arr[x])
            elif type==2:
                X_nhrr.append(arr[0:x])
                y_nhrr.append(arr[x])
            X1.append(arr[0:x])
            y1.append(arr[x])
        end = gc - 1
        if type==1:
            rang_mnist.append([start, end])
        elif type==2:
            rang_nhrr.append([start,end])
        X1 = np.array(X1)
        return [X1, y1]



def conv_data_apply_pca(X1,k):
    mean = np.mean(X1, axis=0)
    normalized = X1 - mean
    cov_matrix = np.cov(normalized.T)
    eigen_values, eigen_vectors = linalg.eig(cov_matrix)
    optk = k
    eigen_vectors = eigen_vectors[:, 0:optk]
    conv_data = np.dot(eigen_vectors.T, normalized.T).T
    X1 = conv_data.real
    return [X1,eigen_vectors]
    '''pca=PCA(k);
    pca.fit(X1);
    return  pca.transform(X1)'''


f1=[]
f2=[]

for i in range(10):
    f1.append(open('features_mnist_more_'+str(i)+'.txt','r'))
    f2.append(open('features_more_' + str(i) + '.txt', 'r'))

gc=0
#print(f)
for i in range(0,10):
    read_file(open('features_mnist_more_'+str(i)+'.txt','r'),274,1)

gc=0
for i in range(0,10):
    read_file(open('features_more_'+str(i)+'.txt','r'),494,2)

print(rang_mnist)
print(rang_nhrr)

X_mnist = np.array(X_mnist)
X_test=X_mnist

for dig in range(1):

    y_test=[]
    for i in range(len(X_test)):
        y_test.append(-1)

    for i in range(rang_mnist[dig][0],rang_mnist[dig][1]):
        y_test[i]=1

    for j in range(rang_mnist[dig][0],rang_mnist[dig][1],600):
        x_mnist=X_mnist[j:min(j+600,rang_mnist[dig][1])]
        y_train = [1 for i in range(len(x_mnist))]
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        print(dig,x_mnist.shape)
        [x_mnist,eigen]=conv_data_apply_pca(x_mnist,20)
        print(np.array(eigen).shape)
        print(np.array(X_test).shape)
        new_Xtest=np.dot(X_test,eigen);
        one_class_svm(x_mnist,new_Xtest,y_train,y_test,dig)

