import  numpy as np

a=np.array([1,2])
b=a.transpose()

c1=b*a

k=np.array([[1,2],[1,2]])

c=a*b*k

print(c1)
print(c)