import PIL
from PIL import Image
import numpy as np
xx=[0,1,2,3,4,5,6,7,8,9]
for X in xx:
    thefile = open('features_mnist'+str(X)+'.txt', 'w')
    print(X)
    try:
        for k in range(0,7000,1):
            try:
                img = Image.open('MNIST_'+str(X)+'/'+str(X)+'_'+str(k)+'.jpeg')
            except Exception as e:
                print(e)
                break
            w,h=img.size
            print(w,h)
            arr = np.array(img)
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
            thefile.write(",".join(map(lambda x: str(x), a)))
            thefile.write(","+str(X)+"\n");
            #print(len(a))
        thefile.close()
    except Exception as e:
        print(e)