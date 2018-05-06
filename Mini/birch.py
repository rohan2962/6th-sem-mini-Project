from sklearn.cluster import Birch
fp=open('Dataset.txt',"r")
X=[]
for line in fp:
    a=0
    b=0
    flag=0
    for i in line:
       if i==' ' and a>0:
           flag=1
       c=i
       if c>='0' and c<='9':
        if flag==0:
            a*=10
            a+=int(c)
        else:
            b*=10
            b+=int(c)

    #print(a,b)

    X.append([a,b])

print('****')
brc = Birch(branching_factor=5, n_clusters=3, threshold=0.5,
compute_labels=True)
brc.fit(X)
print(brc.predict(X[0:200]))