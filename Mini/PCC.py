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
    clf = svm.OneClassSVM(kernel='rbf', gamma=0.5, nu=0.1)
    clf.fit(X_train, y_train)
    print('Built in classifiers accuracy is ' + str(accuracy_score(y_test, clf.predict(X_test)) * 100) + '%')
    #print()
    return

X=[]
X_nhrr=[]
rang=[]
rang_nhrr=[]

def read_file(fp,x,type):
        start = -1
        end = -1
        gc=0
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

#print(f)
for i in range(0,10):
    read_file(open('features_mnist_more_'+str(i)+'.txt','r'),274,1)


for i in range(0,10):
    read_file(open('features_more_'+str(i)+'.txt','r'),494,2)

c=0
for dig in range(10):
    X1=[]
    X2=[]
    y1=[]
    y2=[]
    rang1=[]
    rang2=[]

    X1=X[rang[dig][0]:rang[dig][1]]


    X2=X_nhrr[rang_nhrr[dig][0]:rang_nhrr[dig][1]]




    X1=conv_data_apply_pca(X1,20)
    X2=conv_data_apply_pca(X2,20)

    it1=0
    X2_r=[]
    rohan=[]
    for it2 in range(9):

        X1_r=np.array(X1[it1:it1+600])

        if it1==0:
            X2_r=np.array(X2[0:600])

        f1=len(X1_r[0])
        f2=len(X2_r[0])

        nx1r=X1_r
        nx2r=X2_r

        X1r1=X1_r
        y1r1=[1 for i in range(len(X1_r))]


        a1=[1 for i in range(f1)]
        a2=[1 for i in range(f2)]

        for i in range(f1):
            for j in range(f2):
                if a1[i]==0 or a2[j]==0:
                    continue
                cv1 = np.std(X1_r.T[i])
                cv2 = np.std(X2_r.T[j])
                if cv1==0 or cv2==0:
                    continue
                pcc=np.corrcoef(X1_r.T[i],X2_r.T[j])[0][1]
                pcc=np.cov(X1_r.T[i],X2_r.T[j])[0][1]/(cv1*cv2)
                #print(i,j,pcc,np.corrcoef(X1_r.T[i],X2_r.T[j])[0][1])
                if pcc*10 >= 0.6:
                    if cv1>=cv2:
                        a2[j]=0
                    else:
                        a1[i]=0

        #print(a1)
        #print(a2)

        cnt=0
        for i in a1:
            if i==0:
                cnt=cnt+1
        for i in a2:
            if i==0:
                cnt=cnt+1


        Xp = conv_data_apply_pca(X,len(a1)+len(a2)-cnt)


        Xrtest1 = conv_data_apply_pca(X,20)

        Xrtest2 = conv_data_apply_pca(X_nhrr,20)

        Xp_nhrr=conv_data_apply_pca(X_nhrr,len(a1)+len(a2)-cnt)

        #correlation_matrix=np.corrcoef(X1,X2)

        #print(X1.shape)
        #print(X2.shape)
        #print(correlation_matrix.shape)

        X_train=[]
        y_train=[]
        X_test=[]
        y_test=[]


        for i in range(f1):
            if a1[i]==1:
                X_train.append(X1_r.T[i])

        for i in range(f2):
            if a2[i]==1:
                X_train.append(X2_r.T[i])





        X_train=np.array(X_train)
        X_train=X_train.T

        X2_r=X_train

        y_train=[1 for i in range(len(X_train))]

        X_test=np.concatenate([Xp,Xp_nhrr])

        X_testr=np.concatenate([Xrtest1,Xrtest2])

        #dig=1
        for i in range(0,len(rang)):
            for j in range(rang[i][0],rang[i][1]+1):
                if i==dig:
                    y_test.append(1)
                else:
                    y_test.append(-1)

        for i in range(0, len(rang_nhrr)):
            for j in range(rang_nhrr[i][0], rang_nhrr[i][1] + 1):
                if i == dig:
                    y_test.append(1)
                else:
                    y_test.append(-1)


        print('For digit ',str(dig),'Iteration ',str(it2+1))
        one_class_svm(X1r1,X_testr,y1r1,y_test,dig)
        one_class_svm(X_train,X_test,y_train,y_test,dig)
        print()
        print()
        it1=it1+600






