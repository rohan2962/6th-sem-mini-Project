import PIL
from PIL import Image
import numpy as np
xx=[0,1,2,3,4,5,6,7,8,9]
for X in xx:
    thefile = open('features_mnist_more_'+str(X)+'.txt', 'w')
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
            #print(len(a))
        thefile.close()
    except Exception as e:
        print(e)