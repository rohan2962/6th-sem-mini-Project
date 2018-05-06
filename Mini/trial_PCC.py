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


def one_class_svm(X_train,X_test,y_train,y_test,digit):
    print('For digit '+str(digit))
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    print(X_train.shape,y_train.shape)
    clf = svm.OneClassSVM(kernel='rbf', gamma=0.5, nu=0.1)
    clf.fit(X_train, y_train)
    print('Built in classifiers accuracy is ' + str(accuracy_score(y_test, clf.predict(X_test)) * 100) + '%')
    #print()
    return

X=[]
X_nhrr=[]
rang=[]
rang_nhrr=[]
xxxx=0
def read_file(fp,x,type):
        global gc,xxxx
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
                    if (int(s) < 0):
                        xxxx = xxxx + 1
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
                X.append(arr[0:x])
            elif type==2:
                X_nhrr.append(arr[0:x])
            X1.append(arr[0:x])
            y1.append(arr[x])
        end = gc - 1
        if type==1:
            rang.append([start, end])
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

    # print(optk)
    eigen_vectors = eigen_vectors[:, 0:optk]
    #print(eigen_vectors)
    #print()
    #print()
    #print(eigen_values[0:k])
    conv_data = np.dot(eigen_vectors.T, normalized.T).T
    X1 = conv_data.real
    return  X1

f=[]


for i in range(10):
    f.append(open('features_mnist_more_'+str(i)+'.txt','r'))
    f.append(open('features_more_' + str(i) + '.txt', 'r'))

gc=0
#print(f)
for i in range(0,10):
    read_file(open('features_mnist_more_'+str(i)+'.txt','r'),274,1)

gc=0
for x in range(10):
    for i in range(0,10):
        read_file(open('features_more_'+str(i)+'.txt','r'),494,2,600)
print(xxxx)
print(rang)
print(rang_nhrr)
c=0
Xp = []
Xp_nhrr = []


for dig in range(10):
    if dig==0:
        Xp=X[rang[dig][0]:rang[dig][1]+1]
        Xp_nhrr=X_nhrr[rang_nhrr[dig][0]:rang_nhrr[dig][1]+1]
    else:
        Xp=np.concatenate([Xp,X[rang[dig][0]:rang[dig][1]+1]])
        Xp_nhrr=np.concatenate([Xp_nhrr,X_nhrr[rang_nhrr[dig][0]:rang_nhrr[dig][1]+1]])

for dig in range(10):
    X1=[]
    X2=[]
    y1=[]
    y2=[]
    rang1=[]
    rang2=[]

    X1=X[rang[dig][0]:rang[dig][1]]

    print(rang[dig][0],rang[dig][1])


    X2=X_nhrr[rang_nhrr[dig][0]:rang_nhrr[dig][1]]




    #X1=conv_data_apply_pca(X1,20)
    #X2=conv_data_apply_pca(X2,20)

    it1=0
    X2_r=[]
    rohan=[]
    for it2 in range(9):

        X1_r=np.array(X1[it1:it1+600])
        #X1_r=conv_data_apply_pca(X1_r,25)
        print(it1,it1+600)

        f1=len(X1_r[0])

        nx1r=X1_r
        nx2r=X2_r

        X1r1=X1_r
        y1r1=[1 for i in range(len(X1_r))]




        #correlation_matrix=np.corrcoef(X1,X2)

        #print(X1.shape)
        #print(X2.shape)
        #print(correlation_matrix.shape)

        X_train=[]
        y_train=[]
        X_test=[]
        y_test=[]


        X_test=Xp

        #dig=1
        for i in range(0,len(rang)):
            for j in range(rang[i][0],rang[i][1]+1):
                if i==dig:
                    y_test.append(1)
                else:
                    y_test.append(-1)

        '''for i in range(0, len(rang_nhrr)):
            for j in range(rang_nhrr[i][0], rang_nhrr[i][1] + 1):
                if i == dig:
                    y_test.append(1)
                else:
                    y_test.append(-1)
        '''

        print('For digit ',str(dig),'Iteration ',str(it2+1))
        one_class_svm(X1r1,X_test,y1r1,y_test,dig)
        print()
        print()
        it1=it1+60






