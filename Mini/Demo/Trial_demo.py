import PIL
from PIL import Image
import numpy as np
import numpy.linalg as linalg
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
from sklearn.externals import joblib
from sklearn import svm
import copy
xx=4
data=[]
for X in range(100):
    thefile = open('features_demo'+str(X)+'.txt', 'w')
    print(X)
    try:
        for k in range(0,1,1):
            ls=[]
            try:
                img = Image.open('MNIST_8/8_'+str(X)+'.jpeg')
                
            except Exception as e:
                print(e)
                break
            w,h=img.size
            print(w,h)
            img=img.convert('L')

            img = img.resize((28,28), Image.ANTIALIAS)

            img.save('new_img'+str(X+1)+'.jpg')
            arr = np.array(img)
            #print(arr.)
            a=[]
            '''for i in range(0,50):
                for j in range(0,50):
                    #arr[i][j]/=255;
                    print(arr[i][j],end=' ')
                print()
            '''
            for i in range(0,28):
                f=0
                for j in range(0,28):
                    if arr[i][j] == 0:
                        a.append(j)
                        f=1
                        break
                if f==0:
                    a.append(-1)

            for j in range(0, 28):
                f = 0
                for i in range(27, -1,-1):
                    if arr[i][j] == 0:
                        a.append(i)
                        f = 1
                        break
                if f == 0:
                    a.append(-1)
            for i in range(27, -1,-1):
                f = 0
                for j in range(27,-1,-1):
                    if arr[i][j] == 0:
                        a.append(j)
                        f = 1
                        break
                if f == 0:
                    a.append(-1)
            for j in range(27, -1,-1):
                f = 0
                for i in range(0, 28):
                    if arr[i][j] == 0:
                        a.append(i)
                        f = 1
                        break
                if f == 0:
                    a.append(-1)
            for i in range(0,28):
                max = 0
                count=0
                for j in range(0,28):
                    if j==0 and arr[i][j] == 0:
                        count = count+1;
                        if(count>max):
                            max=count
                    elif arr[i][j] == 0 and arr[i][j-1] == 0 :
                        count = count+1
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i][j-1] == 1 :
                        count = 1
                        if (count > max):
                            max = count
                a.append(max)
            print('3', end='')
            for i in range(0,28):
                max = 0
                count=0
                for j in range(0,28):
                    if j==0 and arr[j][i] == 0:
                        count = count+1;
                        if(count>max):
                            max=count
                    elif arr[j][i] == 0 and arr[j-1][i] == 0 :
                        count = count+1
                        if (count > max):
                            max = count
                    elif arr[j][i] == 0 and arr[j-1][i] == 1 :
                        count = 1
                        if (count > max):
                            max = count
                a.append(max)
            print('4', end='')
            for k in range(0,27,1):
                i=k
                j=0
                max=0
                count=0
                while(i<28):
                    if j==0 and arr[i][j] == 0:
                        count = count+1;
                        if(count>max):
                            max=count
                    elif arr[i][j] == 0 and arr[i-1][j-1] == 0 :
                        count = count+1
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i-1][j-1] == 1 :
                        count = 1
                        if (count > max):
                            max = count
                    i = i+1
                    j = j+1
                a.append(max)
            print('5', end='')
            for k in range(1,27,1):
                j=k
                i=0
                max=0
                count=0
                while(j<28):
                    if i==0 and arr[i][j] == 0:
                        count = count+1;
                        if(count>max):
                            max=count
                    elif arr[i][j] == 0 and arr[i-1][j-1] == 0 :
                        count = count+1
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i-1][j-1] == 1 :
                        count = 1
                        if (count > max):
                            max = count
                    i = i+1
                    j = j+1
                a.append(max)
            print('6', end='')
            for k in range(0,27,1):
                i=k
                j=27
                max=0
                count=0
                while(i<28):
                    if j==27 and arr[i][j] == 0:
                        count = count+1;
                        if(count>max):
                            max=count
                    elif arr[i][j] == 0 and arr[i-1][j+1] == 0 :
                        count = count+1
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i-1][j+1] == 1 :
                        count = 1
                        if (count > max):
                            max = count
                    i = i+1
                    j = j-1
                a.append(max)
            print('7',end='')
            for k in range(26,0,-1):
                j=k
                i=0
                max=0
                count=0
                while(j>=0):
                    if i==0 and arr[i][j] == 0:
                        count = count+1;
                        if(count>max):
                            max=count
                    elif arr[i][j] == 0 and arr[i-1][j+1] == 0 :
                        count = count+1
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i-1][j+1] == 1 :
                        count = 1
                        if (count > max):
                            max = count
                    i = i+1
                    j = j-1
                a.append(max)
            thefile.write(",".join(map(lambda x: str(x), a)))
            thefile.write(","+str(X)+"\n");
            data.append(a)
            #print(len(a))
        thefile.close()
    except Exception as e:
        print(e)

print('***',len(data),len(data[0]))
#print(data[0][30:35])
k3=[]


eigen_vec=np.load('eigen_vec.npy')

f=open('k3.txt',"r")
for i in f:
    k3.append(int(i))


data=np.array(data)
for j in range(0,10):
    #clf = svm.OneClassSVM(kernel='rbf', gamma=0.006, nu=0.2)
    clf=joblib.load('Model'+str(j)+'.pkl')

    #print(clf.support_vectors_)
    eigen=eigen_vec[j]
    dk3=len(eigen[0])
    print(dk3,k3[j])
    '''mean = np.mean(data, axis=0)
    normalized = data - mean
    cov_matrix = np.cov(normalized.T)
    eigen_values, eigen_vectors = linalg.eig(cov_matrix)
    eigen_values=np.real(eigen_values)
    eigen_vectors=np.real(eigen_vectors)
    #print(np.array(cov_matrix))
    #print(eigen_vectors.shape)
    optk = dk3
    eigen_vectors = eigen_vectors[:, 0:optk]
    #print(eigen_vectors.shape)
    #print()
    #print()
    #print(eigen_values[0:20])'''
    pca=PCA(n_components=dk3)
    new_data=pca.fit_transform(data)
    print(np.array(new_data).shape)
    #new_data = np.dot(eigen_vectors.T, normalized.T).T
    #print(np.array(eigen).shape,np.array(new_data).shape)
    new_data=np.dot(eigen,new_data.T).T
    #print(dk3,np.array(new_data).shape)
    res = clf.predict(new_data)
    print(res)
