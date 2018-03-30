from sklearn import svm
import cvxpy as cvx
import numpy as np
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

f0 = open('features_0.txt','r')
f6 = open('features_6.txt','r')
f8 = open('features_8.txt','r')
f9 = open('features_9.txt','r')
X0,y0,X6,y6,X8,y8,X9,y9=[],[],[],[],[],[],[],[]
for pp in[0,6,8,9]:
    X,y=[],[]
    fp=0
    if pp==0:
        fp=f0
    elif pp==6:
        fp=f6
    elif pp==8:
        fp=f8
    elif pp==9:
        fp=f9
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
    if pp == 0:
        X0,y0=X,y
    elif pp == 6:
        X6, y6 = X, y
    elif pp == 8:
        X8,y8=X,y
    elif pp == 9:
        X9,y9=X,y
    #print(y)
clf=svm.OneClassSVM(kernel = 'rbf', gamma = 0.0001, nu = 0.01)
#clf=svm.SVC()
X=[]
X_train=[]
X_test=[]


j=0
for i in X0:
    if j<=500:
        X_train.append(i)
    j=j+1
    #X.append(i)
    X_test.append(i)
for i in X6:
    X_test.append(i)
    #X_train.append(i)
    X.append(i)
for i in X8:
    #X_train.append(i)
    X_test.append(i)
    X.append(i)
for i in X9:
    #X_train.append(i)
    X_test.append(i)
    X.append(i)


print('****'+str(len(X_train)))

X=np.array(X)
print(X.shape)
y=[]
y_train,y_test=[],[]
print(np.array(X_train).shape,np.array(X_test).shape)
for i in y0:
    y_train.append(1.0)
    y_test.append(1.0)
    y.append(1)
for i in y6:
    y.append(0)
    #y_train.append(1.0)
    y_test.append(-1.0)
for i in y8:
    y.append(0)
    #y_train.append(1.0)
    y_test.append(-1.0)
for i in y9:
    y.append(0)
    #y_train.append(1.0)
    y_test.append(-1.0)

y=np.array(y)

y.reshape([-1,])
print(y.shape)
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
clf.fit(X_train,y_train)
print(accuracy_score(y_test,clf.predict(X_test)))

alpha1=cvx.Variable(len(X_train),1)
K1=[]
for i in X_train:
    K1.append(cal(i,i))
K1=np.array(K1)
K1=K1.reshape(-1,1)
print(K1.shape)
#alpha1=np.array(alpha1)

#print(cvx.sum_entries(alpha1*K1))




K2=[]
for i in X_train:
    tp=[]
    for j in X_train:
        tp.append(cal(i,j))
    K2.append(tp)



P = 2*cvxopt.matrix(K2)
#print(P.size)
q=-1*cvxopt.matrix(K1)
#print(q.size)
A=[1.0 for i in range(len(X_train))]
A=cvxopt.matrix(A)
A=A.T
#print(A.size)
b=cvxopt.matrix(1.0)
#h=cvxopt.matrix(1.0)
h=[1.0 for i in range(len(X_train))]
G=[[1.0 for i in range(len(X_train))] for i in range(len(X_train))]
for i in range(len(X_train)):
    for j in range(len(X_train)):
        if i==j:
            G[i][j]=1.0
        else:
            G[i][j]=0.0

G=cvxopt.matrix(G)
h=cvxopt.matrix(h)

sol=cvxopt.solvers.qp(P, q, G, h, A, b)



sx=sol['x']
print(sx)

center=[0 for i in range(len(X_train[0]))]
for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        center[j]+=(sx[j]*X_train[i][j])
radius=0.0


for i in X_train:
    radius+=dis(i,center)
radius/=len(X_train)


acc=0.0
for i in range(len(X_test)):
    if dis(center,X_test[i])<=radius and y_test[i]==1.0:
        acc+=1.0
    elif dis(center,X_test[i])>radius and y_test[i]==-1.0:
        acc+=1.0

print(acc/len(X_test))

