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
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


def one_class_svm(X_train, X_test, y_train, y_test, digit, ):
    #print('For digit ' + str(digit))
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    #print(X_train.shape, y_train.shape)

    '''C1 = np.arange(0.001,0.09,0.001)


    C2 = np.arange(0.1,0.9,0.1)

    #0.1 0.9 0.1
    #0.001 0.1 0.003

    for gamma in C1:
        for nu in C2:
            clf = svm.OneClassSVM(kernel='rbf',gamma=gamma,nu=nu)
            clf.fit(X_train,y_train)
            print(gamma,nu,accuracy_score(y_test, clf.predict(X_test)),accuracy_score(y_train, clf.predict(X_train)))


    '''

    clf = svm.OneClassSVM(kernel='rbf', gamma=0.006, nu=0.2)

    clf.fit(X_train)
    res_self = clf.predict(X_train)
    res = clf.predict(X_test)
    #print('score for class ', digit, np.average(res_self == y_train))
    #print('Built in classifiers accuracy is ' + str(accuracy_score(y_test, clf.predict(X_test)) * 100) + '%')
    # print()
    #joblib.dump(clf, 'Model' + str(digit) + '.pkl')
    # clf=joblib.load('Model'+str(digit)+'.pkl')
    '''res=clf.predict(X_train)
    cnt=0
    for i in res:
        if i==1:
            cnt=cnt+1
    print('*****',cnt)'''
    return (accuracy_score(y_test, clf.predict(X_test)) * 100+np.average(res_self == y_train) * 100)/2.0


X_mnist = []
X_nhrr = []
y_mnist = []
y_nhrr = []
rang_mnist = []
rang_nhrr = []


def read_file(fp, x, type):
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
        if type == 1:
            X_mnist.append(arr[0:x])
            y_mnist.append(arr[x])
        elif type == 2:
            X_nhrr.append(arr[0:x])
            y_nhrr.append(arr[x])
        X1.append(arr[0:x])
        y1.append(arr[x])
    end = gc - 1
    if type == 1:
        rang_mnist.append([start, end])
    elif type == 2:
        rang_nhrr.append([start, end])
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
    optk = 0
    for k in range(1, len(X1[0])):
        pca = PCA(k);
        pca.fit(X1);
        if np.sum(pca.explained_variance_ratio_) > 0.95:
            #print('opt is ', k)
            optk = k;
            break
    return [pca.transform(X1), pca.components_, optk]


def conv_pca(X, k):
    pca = PCA(k);
    pca.fit(X);
    return pca.transform(X)


f1 = []
f2 = []

for i in range(10):
    f1.append(open('features_mnist_more_' + str(i) + '.txt', 'r'))
    f2.append(open('features_more_' + str(i) + '.txt', 'r'))

gc = 0
# print(f)
for i in range(0, 10):
    read_file(open('features_mnist_more_' + str(i) + '.txt', 'r'), 274, 1)

gc = 0
for i in range(0, 10):
    read_file(open('features_more_' + str(i) + '.txt', 'r'), 494, 2)

print(rang_mnist)
#print(rang_nhrr)

scaler = StandardScaler()
# X_nhrr=scaler.fit_transform(X_nhrr)


X_mnist = np.array(X_mnist)
X_test = X_mnist

#f2 = open('k3.txt', "w+")
global_eig = []
for dig in [0,1,6,9]:
    x_mnist1 = X_mnist[0:min(0 + 600, rang_mnist[dig][1])]
    [x_mnist1, eigen, k1] = conv_data_apply_pca(x_mnist1)
    no_of_fe = []
    acc = []
    k_2=[]
    k_4 = []

    it = 0
    final_eig = []
    final_k3 = 0
    for j in range(rang_mnist[dig][0], rang_mnist[dig][1], 600):
        x_mnist = X_mnist[j + 600:min(j + 1200, rang_mnist[dig][1])]
        if len(x_mnist) < 600:
            continue
        y_train = [1 for i in range(len(x_mnist))]
        y_train = np.array(y_train)
        # print(dig,x_mnist.shape)

        [x_mnist, eigen, k2] = conv_data_apply_pca(x_mnist)
        k_2.append(k2)
        # print(np.array(eigen).shape)
        # print(np.array(X_test).shape)

        # new_Xtest=np.dot(X_test,eigen);
        # one_class_svm(x_mnist,new_Xtest,y_train,y_test,dig)
        #if it == 0:
        #    [x_mnist1, eigen, k2] = conv_data_apply_pca(x_mnist1)

        l1 = len(x_mnist1[0]);
        l2 = len(x_mnist[0]);
        a1 = [1 for i in range(l1)]
        a2 = [1 for i in range(l2)]
        #print(l1, l2)
        for i in range(l1):
            for j in range(l2):
                if a1[i] == 0 or a2[j] == 0:
                    continue
                cv1 = np.std(x_mnist1.T[i])
                cv2 = np.std(x_mnist.T[j])
                if cv1 == 0 or cv2 == 0:
                    continue
                pcc = np.corrcoef(x_mnist1.T[i], x_mnist.T[j])[0][1]
                # print(pcc)
                if pcc  >= 0.6:
                    if cv1 >= cv2:
                        a2[j] = 0
                    else:
                        a1[i] = 0

        new_x = []
        for i in range(l1):
            if a1[i] == 1:
                tmp = []
                for j in x_mnist1:
                    tmp.append(j[i])
                new_x.append(tmp)

        for i in range(l2):
            if a2[i] == 1:
                tmp = []
                for j in x_mnist:
                    tmp.append(j[i])
                new_x.append(tmp)

        # print(len(new_x[0]))
        new_x = np.array(new_x)
        new_x = new_x.T
        # print(len(new_x[0]),'pcc ***')
        # print(new_x.shape)

        x_mnist1 = []

        for i in new_x:
            x_mnist1.append(i)

        x_mnist1 = np.array(x_mnist1)

        [x_mnist1, eigen, k4] = conv_data_apply_pca(x_mnist1)
        #print('**** req',k4)

        k3 = len(new_x[0])

        new_Xtest = []
        y_test = []
        cnt = 0
        xlc = conv_pca(X_test, k3)
        xlc = np.dot(eigen, xlc.T).T
        final_eig = copy.deepcopy(eigen)
        final_k3 = k3
        for i in rang_mnist:
            xlc1 = xlc[i[0]:i[1]]
            # xlc=conv_pca(X_test[i[0]:i[1]],k3)
            # xlc=np.dot(eigen,xlc.T).T
            # print(eigen.shape,xlc.shape)
            for j in xlc1:
                if cnt == dig:
                    y_test.append(1)
                else:
                    y_test.append(-1)
                new_Xtest.append(j)
            cnt = cnt + 1
        # print(np.array(new_Xtest).shape)
        no_of_fe.append(k4)
        ac1 = one_class_svm(x_mnist1, new_Xtest, y_train, y_test, dig)
        acc.append(ac1)
        k_4.append(k4)
        it = it + 1
        print('At batch no '+str(it)+' for digit ' +str(dig)+'  '+str(k1)+' and '+str(k2)+' features are merged and after merging no of features is '+str(k4)+' and Accuracy obtained is '+str(ac1))
        k1=k4

    print()
    print()
    #print('Final feature req for digit '+str(dig)+' is '+str(k4))
    #print(len(final_eig[0]))
    global_eig.append(final_eig)

    #f2.write(str(final_k3))
    #f2.write("\n")
    #print(acc)
    plt.plot([i+1 for i in range(len(acc))],acc,marker='o')
    plt.xlabel('No of batches merged')
    plt.ylabel('Accuracy')
    #plt.show()
    plt.savefig('Final Acc vs Batches merged for Digit  '+str(dig)+'.png')
    plt.clf()
    print(no_of_fe)
    #print(k_2)
    plt.plot([i for i in range(len(k_4))],k_4,label='No of components after merging',marker='o')
    #print(k_2)
    #plt.plot([i for i in range(len(no_of_fe))], k_2,label='Inital no of components',marker='o')
    #plt.ylim([10, 150])
    plt.xlabel('No of batches merged')
    plt.ylabel('No of features')
    plt.legend()
    #plt.show()
    plt.savefig('Final No of features vs Batches merged for Digit  '+str(dig)+'.png')
    plt.clf()

#print(np.array(global_eig).shape)
#np.save('eigen_vec.npy', np.array(global_eig))
#print('saved')

# p=np.load('eigen_vec.npy')
# print(p)