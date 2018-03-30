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
    clf = svm.OneClassSVM(kernel='rbf', gamma=0.01, nu=0.01)
    clf.fit(X_train, y_train)
    print('Built in classifiers accuracy is ' + str(accuracy_score(y_test, clf.predict(X_test)) * 100) + '%')
    #print()
    return
    alpha1 = cvx.Variable(len(X_train), 1)
    K1 = []
    for i in X_train:
        K1.append(cal(i, i))
    K1 = np.array(K1)
    K1 = K1.reshape(-1, 1)

    K2 = []
    for i in X_train:
        tp = []
        for j in X_train:
            tp.append(cal(i, j))
        K2.append(tp)

    P = 2 * cvxopt.matrix(K2)
    # print(P.size)
    q = -1 * cvxopt.matrix(K1)
    # print(q.size)
    A = [1.0 for i in range(len(X_train))]
    A = cvxopt.matrix(A)
    A = A.T
    # print(A.size)
    b = cvxopt.matrix(1.0)
    # h=cvxopt.matrix(1.0)
    h = [1.0 for i in range(len(X_train))]
    G = [[1.0 for i in range(len(X_train))] for i in range(len(X_train))]
    for i in range(len(X_train)):
        for j in range(len(X_train)):
            if i == j:
                G[i][j] = 1.0
            else:
                G[i][j] = 0.0

    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    print('***')
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)

    sx = sol['x']

    center = [0 for i in range(len(X_train[0]))]
    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            center[j] += (sx[j] * X_train[i][j])
    radius = 0.0

    for i in X_train:
        radius += dis(i, center)
    radius /= len(X_train)

    acc = 0.0
    for i in range(len(X_test)):
        if dis(center, X_test[i]) <= radius and y_test[i] == 1.0:
            acc += 1.0
        elif dis(center, X_test[i]) > radius and y_test[i] == -1.0:
            acc += 1.0
    acc=acc/len(X_test)
    print('Our classifiers accuracy is '+str(acc*100)+'%')
    print()


X=[]
y=[]
f=[]

for i in range(10):
    #X.append([])
    #y.append([])
    f.append(open('features_mnist'+str(i)+'.txt','r'))

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
        X.append(arr[0:112])
        y.append(arr[112])
    end=gc-1
    rang.append([start,end])

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

for i in range(10):
    yi=[-1 for i in range(len(y))]
    yi=np.array(yi);
    yi[rang[i][0]:rang[i][1]+1]=1
    X_train=X[rang[i][0]:rang[i][1]+1-1000]
    X_test=X
    y_train=[1.0 for i in range(rang[i][1]-rang[i][0]+1-1000)]
    y_test=yi
    one_class_svm(X_train,X_test,y_train,y_test,i)


