from sklearn import svm
import cvxpy as cvx
import numpy as np
import copy
import math
import numpy.linalg as linalg
from sklearn.cross_validation import train_test_split
import pandas as pd
import math
import cvxopt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def one_class_svm(X_train,X_test,y_train,y_test,digit,bn):
    print('For digit '+str(digit))
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    print(X_train.shape,y_train.shape)

    C1 = np.arange(0.001,0.09,0.001)


    C2 = np.arange(0.1,0.9,0.1)

    #0.1 0.9 0.1
    #0.001 0.1 0.003
    if bn==1222:
        for gamma in C1:
            for nu in C2:
                clf = svm.OneClassSVM(kernel='rbf',gamma=gamma,nu=nu)
                clf.fit(X_train,y_train)
                if accuracy_score(y_train, clf.predict(X_train))>=0.7:
                    print(gamma,nu,accuracy_score(y_test, clf.predict(X_test)),accuracy_score(y_train, clf.predict(X_train)))
    #clf = svm.OneClassSVM(kernel='rbf')

    '''C1 = np.arange(0.001,0.1,0.003);
    C2 = np.arange(0.1,0.9,0.1);
    #0.1 0.9 0.1
    #0.001 0.1 0.003

    clf1 = GridSearchCV(estimator=clf, param_grid=dict(gamma=C1,nu=C2), scoring='accuracy')
    clf1.fit(X_train,y_train)
    print(clf1.best_score_)
    params = clf1.best_params_
    gamma = params['gamma']
    nu = params['nu']'''
    clf=svm.OneClassSVM(kernel='rbf',gamma=0.088,nu=0.4)
    clf.fit(X_train)
    res_self = clf.predict(X_train)
    res = clf.predict(X_test)
    print('score for class ',digit, np.average(res_self == y_train))
    print('Built in classifiers accuracy is ' + str(accuracy_score(y_test, clf.predict(X_test)) * 100) + '%')
    #print()
    return accuracy_score(y_test, clf.predict(X_test)) * 100

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



def conv_data_apply_pca(X1):
    '''mean = np.mean(X1, axis=0)
    normalized = X1 - mean
    cov_matrix = np.cov(normalized.T)
    eigen_values, eigen_vectors = linalg.eig(cov_matrix)
    optk = k
    eigen_vectors = eigen_vectors[:, 0:optk]
    conv_data = np.dot(eigen_vectors.T, normalized.T).T
    X1 = conv_data.real
    return [X1,eigen_vectors]'''
    optk=0
    for k in range(1,len(X1[0])):
        pca=PCA(k);
        pca.fit(X1);
        if np.sum(pca.explained_variance_ratio_)>0.95:
            print('opt is ',k)
            optk=k;
            break
    return  [pca.transform(X1),pca.components_,optk]

def conv_pca(X,k):
    pca=PCA(k);
    pca.fit(X);
    return pca.transform(X)


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



scaler=StandardScaler()
#X_mnist=scaler.fit_transform(X_mnist)
#X_nhrr=scaler.fit_transform(X_nhrr)


X_mnist = np.array(X_mnist)
X_test=X_mnist

for dig in [1,2,3]:

    no_of_fe=[]
    acc=[]
    x_nhrr=X_nhrr[rang_nhrr[dig][0]:min(rang_nhrr[dig][0]+600,rang_nhrr[dig][1])]


    it=0

    for j in range(rang_mnist[dig][0],rang_mnist[dig][1],600):
        x_mnist=X_mnist[j:min(j+600,rang_mnist[dig][1])]
        if len(x_mnist)<600:
            continue
        y_train = [1 for i in range(len(x_mnist))]
        y_train = np.array(y_train)
        print(dig,x_mnist.shape)

        [x_mnist,eigen,k1]=conv_data_apply_pca(x_mnist)

        print(np.array(eigen).shape)
        print(np.array(X_test).shape)

        #new_Xtest=np.dot(X_test,eigen);
        #one_class_svm(x_mnist,new_Xtest,y_train,y_test,dig)
        if it==0:
            [x_nhrr,eigen,k2] = conv_data_apply_pca(x_nhrr)

        l1=len(x_nhrr[0]);
        l2=len(x_mnist[0]);
        a1=[1 for i in range(l1)]
        a2=[1 for i in range(l2)]
        print(l1,l2)
        for i in range(l1):
            for j in range(l2):
                if a1[i]==0 or a2[j]==0:
                    continue
                cv1 = np.std(x_nhrr.T[i])
                cv2 = np.std(x_mnist.T[j])
                if cv1==0 or cv2==0:
                    continue
                pcc=np.corrcoef(x_nhrr.T[i],x_mnist.T[j])[0][1]
                #print(pcc)
                if pcc*10 >= 0.7:
                    if cv1>=cv2:
                        a2[j]=0
                    else:
                        a1[i]=0

        new_x=[]
        for i in range(l1):
            if a1[i]==1:
                tmp=[]
                for j in x_nhrr:
                    tmp.append(j[i])
                new_x.append(tmp)

        for i in range(l2):
            if a2[i]==1:
                tmp=[]
                for j in x_mnist:
                    tmp.append(j[i])
                new_x.append(tmp)

        print(len(new_x[0]))
        new_x=np.array(new_x)
        new_x=new_x.T
        print(len(new_x[0]),'pcc ***')
        print(new_x.shape)

        x_nhrr=[]

        for i in new_x:
            x_nhrr.append(i)

        x_nhrr=np.array(x_nhrr)

        [x_nhrr,eigen,k3]=conv_data_apply_pca(x_nhrr)

        k3=len(new_x[0])



        new_Xtest=[]
        y_test=[]
        cnt=0
        for i in rang_mnist:
            xlc=conv_pca(X_test[i[0]:i[1]],k3)
            xlc=np.dot(eigen,xlc.T).T
            print(eigen.shape,xlc.shape)
            for j in xlc:
                if cnt==dig:
                    y_test.append(1)
                else:
                    y_test.append(-1)
                new_Xtest.append(j)
            cnt=cnt+1
        print(np.array(new_Xtest).shape)
        acc.append(one_class_svm(x_nhrr,new_Xtest,y_train,y_test,dig,it))
        no_of_fe.append(len(x_nhrr[0]))
        it=it+1
    plt.plot([i for i in range(10)],no_of_fe)
    plt.show()
