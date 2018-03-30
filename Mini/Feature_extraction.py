import PIL
from PIL import Image
xx=[0,1,2,3,4,5,6,7,8,9]
for X in xx:
    thefile = open('features_'+str(X)+'.txt', 'w')
    print(X)
    try:
        for k in range(1,800,1):
            try:
                img = Image.open('Last_'+str(X)+'/'+str(X)+'_('+str(k)+').jpeg')
            except Exception as e:
                print(e)
                continue
            w,h=img.size
            print(w,h)
            import numpy as np
            arr = np.array(img)

            arr[arr<200]=0;
            arr[arr>=200]=1;
            a=[]
            '''for i in range(0,50):
                for j in range(0,50):
                    #arr[i][j]/=255;
                    print(arr[i][j],end=' ')
                print()
            '''
            for i in range(0,50):
                f=0
                for j in range(0,50):
                    if arr[i][j] == 0:
                        a.append(j)
                        f=1
                        break
                if f==0:
                    a.append(-1)

            for j in range(0, 50):
                f = 0
                for i in range(49, -1,-1):
                    if arr[i][j] == 0:
                        a.append(i)
                        f = 1
                        break
                if f == 0:
                    a.append(-1)
            for i in range(49, -1,-1):
                f = 0
                for j in range(49,-1,-1):
                    if arr[i][j] == 0:
                        a.append(j)
                        f = 1
                        break
                if f == 0:
                    a.append(-1)
            for j in range(49, -1,-1):
                f = 0
                for i in range(0, 50):
                    if arr[i][j] == 0:
                        a.append(i)
                        f = 1
                        break
                if f == 0:
                    a.append(-1)
            thefile.write(",".join(map(lambda x: str(x), a)))
            thefile.write(","+str(X)+"\n");
            #print(len(a))
        thefile.close()
    except Exception as e:
        print(e)