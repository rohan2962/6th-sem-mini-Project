import PIL
from PIL import Image
xx=[2,3,4,5,7]
xx=[0,6,8,9]
xx=[1]
for X in xx:
    thefile = open('features_zone_'+str(X)+'.txt', 'w')
    print(X)
    try:
        for k in range(1,800,1):
            try:
                img = Image.open('Last_'+str(X)+'/'+str(X)+'_'+str(k)+'.jpeg')
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
            for i1 in range(0,50,10):
                for j1 in range(0,50,10):
                    new_matrix=[]
                    for i in range(0,10):
                        vec=[]
                        for j in range(0,10):
                            vec.append(arr[i+i1][j+j1])
                        new_matrix.append(vec)
                    for i in range(0,10):
                        f=0
                        for j in range(0,10):
                            if new_matrix[i][j] == 0:
                                a.append(j)
                                f=1
                                break
                        if f==0:
                            a.append(-1)

                    for j in range(0, 10):
                        f = 0
                        for i in range(9, -1,-1):
                            if new_matrix[i][j] == 0:
                                a.append(i)
                                f = 1
                                break
                        if f == 0:
                            a.append(-1)
                    for i in range(9, -1,-1):
                        f = 0
                        for j in range(9,-1,-1):
                            if new_matrix[i][j] == 0:
                                a.append(j)
                                f = 1
                                break
                        if f == 0:
                            a.append(-1)
                    for j in range(9, -1,-1):
                        f = 0
                        for i in range(0, 10):
                            if new_matrix[i][j] == 0:
                                a.append(i)
                                f = 1
                                break
                        if f == 0:
                            a.append(-1)
            print(len(a))
            thefile.write(",".join(map(lambda x: str(x), a)))
            thefile.write(","+str(X)+"\n");
            #print(len(a))
        thefile.close()
    except Exception as e:
        print(e)