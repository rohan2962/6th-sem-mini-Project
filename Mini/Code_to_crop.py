from scipy.misc import imsave
from PIL import Image
#thefile = open('features_0.txt', 'w')
for k in range(1,607,1):
    img = Image.open('7_50by50/7_'+str(k)+'.jpeg')
    w,h=img.size
    print(w,h)
    import numpy as np
    arr = np.array(img)
    #arr.reshape()
    arr[arr<200]=0;
    arr[arr>=200]=1;
    #if k==65:
     #   arr[40][0]=arr[40][1]=arr[41][0]=arr[41][1]=1
    a=[]
    g = 0
    for i in range(0,50):
        f=0

        for j in range(0,50):
            if arr[i][j] == 0:
                f=1
                break
        if f==1:
            if g==0:
                g=1
                a1=i
        elif g==1:
            a.append([a1,i-1])
            g=0

    g = 0
    for j in range(0, 50):
        f = 0
        for i in range(49, -1,-1):
            if arr[i][j] == 0:
                f = 1
                break
        if f == 1:
            if g == 0:
                g = 1
                a1 = j
        elif g == 1:
            a.append([a1,j- 1])
            g = 0
    g=0
    b=[]
    try:
        for i in range(a[0][0]-2,a[0][1]+3):
            b1=[]
            if i>= a[0][0] and i <= a[0][1]:
                for j in range(a[1][0]-2,a[1][1]+3):
                    if j >=a[1][0] and j <= a[1][1]:
                        b1.append(arr[i][j])
                    else:
                        b1.append(1)
            else:
                for j in range(0,a[1][1]-a[1][0]+5):
                    b1.append(1)
            b.append(b1)
        b=np.array(b)
        #print(b.shape)

        imsave('Crop_7/7_'+str(k)+'.jpeg',b)
    except  Exception as e:
         print(e)

#thefile.close()