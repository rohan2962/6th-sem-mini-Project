import numpy as np

x=[[1,2],[3,4],[5,6]]

x=np.array(x)
x=x.T

print(x[1])

m1=np.mean(x[0])
m2=np.mean(x[1])

print(x[0]-m1)
print(m2,x[1]-m2)



print((np.array(x[0]-m1)*np.array(x[1]-m2)))

print()
print()

x=x[:,0:2]

print(x)









