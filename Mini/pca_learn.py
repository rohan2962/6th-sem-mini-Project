import numpy as np
import numpy.linalg as lin
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

def covr(c,i,j):
    c=np.array(c)
    c=c.T
    i=c[i]
    j=c[j]
    return np.sum(np.array(i-np.mean(i))*np.array(j-np.mean(j)))/(len(i)-1)


a=np.array([[1,2], [3,4], [5,6] ,[7,8]])
b=np.array([0,1,2,3])

svm2=svm.SVC()
svm2.fit(a,b)
print(accuracy_score(svm2.predict(a),b))

m=np.mean(a,axis=0)

c=a-m

print(a)
print()
print(c)

b=[]

for i in range(len(c[0])):
    nl=[]
    for j in range(len(c[0])):
        nl.append(covr(c,i,j))
    b.append(nl)

b=np.array(b)

print('Covariance Matrix using built in function')
print(np.cov(c.T))
print()
print('Covariance Matrix using own function')
print(b)


val,vec=lin.eig(b)
print()
print(val)
print()
print(vec)

vec=vec[:,0:1]

conv_data=np.dot(vec.T,a.T)

print()

print(conv_data.T)

print()




print(np.dot(vec,conv_data).T)

svm1=svm.SVC()
b=np.array([0,1,2,3])
svm1.fit(conv_data.T,b)
print(accuracy_score(svm1.predict(conv_data.T),b))
