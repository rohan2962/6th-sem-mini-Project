import numpy as np
from sklearn.preprocessing import StandardScaler

print(np.corrcoef([20,23,8,29,14,11,11,20,17,17],[30,35,21,33,33,26,22,31,33,36])[0][1])

x=[[4,5],[1,2]]
x=np.array(x)
print(x.T[0])

print(np.var([1,2,4]))

a=[[3,3,21],[4,5,4]]
b=[[1,2,3]]
print(np.concatenate([a,b]))

a=[4,5,65]

print(a[0:2])


for i in range(3):
    f=[2,3]

print(f)


s="-12"
print(int(s))

a=[[3,4,-1,51],[42,50,-1,8]]
a=np.array(a)
scaler=StandardScaler()
print(scaler.fit_transform(a))

a=[[1,2,4]]
b=[[5,6,7]]
a=np.array(a)
b=np.array(b)
print(np.hstack((a,b)))

a=[[1,3],[5,6]]
print(a[:][1])