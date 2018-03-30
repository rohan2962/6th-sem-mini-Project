import numpy as np
import csv
from PIL import Image
from scipy.misc import imshow,imsave

data=[]
with open('mnist_train.csv',"r") as file:
    csv_reader=csv.reader(file)
    for row in csv_reader:
        if not row:
            continue
        data.append(row)

print(data[0])
X=[]
y=[]
for row in data:
    y.append(int(row[0]))
    x=[]
    for i in range(1,len(row)):
        x.append(int(row[i]))
    X.append(x)

print(len(X),len(X[0]),X[0])

cnt=[0 for i in range(10)]
nx=X[10]
nx=np.array(nx)
nx=nx.reshape([28,28])
imsave('trial1.jpeg',nx)

for i1 in range(len(X)):
    nx=X[i1]
    nx=np.array(nx)

    np1=np.zeros([28,28])
    c=0
    for i in range(0,28):
        for j in range(0,28):
            if nx[c]>150:
                np1[i][j]=0
            else:
                np1[i][j]=1
            c=c+1
    imsave("MNIST_"+str(y[i1])+"/"+str(y[i1])+"_"+str(cnt[y[i1]])+".jpeg",np1)
    cnt[y[i1]]=cnt[y[i1]]+1



'''for i1 in range(25):
    nx=X[i1]
    nx=np.array(nx)

    np1=np.zeros([28,28])
    c=0
    for i in range(0,28):
        for j in range(0,28):
            if nx[c]>150:
                np1[i][j]=0
            else:
                np1[i][j]=1
            c=c+1
    imsave("trial_"+str(i1)+".jpeg",np1)
    cnt[y[i1]]=cnt[y[i1]]+1

for i1 in range(25):
    nx=X[i1]
    nx=np.array(nx)

    np1=np.zeros([28,28])
    c=0
    for i in range(0,28):
        for j in range(0,28):
            if nx[c]>100:
                np1[i][j]=0
            else:
                np1[i][j]=1
            c=c+1
    imsave("trial_"+str(i1+25)+".jpeg",np1)
    cnt[y[i1]]=cnt[y[i1]]+1
    
'''