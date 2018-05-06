import PIL
from PIL import Image
import numpy as np
import numpy.linalg as linalg
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
from sklearn.externals import joblib
from sklearn import svm
import copy

xx = 4
data = []
for X in range(1,51,1):
    thefile = open('Demo_folder/features_demo' + str(X) + '.txt', 'w+')
    print(X)
    try:
        for k in range(0, 1, 1):
            ls = []
            try:
                #img = Image.open('MNIST_8/8_' + str(X) + '.jpeg')
                img=Image.open('Demo_folder/3_'+str(X)+'.jpeg')
            except Exception as e:
                print(e)
                break
            w, h = img.size
            #print(w, h)
            img = img.convert('L')

            img = img.resize((50, 50), Image.ANTIALIAS)

            img.save('new_img' + str(X + 1) + '.jpg')
            arr = np.array(img)
            # print(arr.)
            a = []
            '''for i in range(0,50):
                for j in range(0,50):
                    #arr[i][j]/=255;
                    print(arr[i][j],end=' ')
                print()
            '''
            for i in range(0, 50):
                f = 0
                for j in range(0, 50):
                    if arr[i][j] == 0:
                        a.append(j)
                        f = 1
                        break
                if f == 0:
                    a.append(-1)

            for j in range(0, 50):
                f = 0
                for i in range(49, -1, -1):
                    if arr[i][j] == 0:
                        a.append(i)
                        f = 1
                        break
                if f == 0:
                    a.append(-1)
            for i in range(49, -1, -1):
                f = 0
                for j in range(49, -1, -1):
                    if arr[i][j] == 0:
                        a.append(j)
                        f = 1
                        break
                if f == 0:
                    a.append(-1)
            for j in range(49, -1, -1):
                f = 0
                for i in range(0, 50):
                    if arr[i][j] == 0:
                        a.append(i)
                        f = 1
                        break
                if f == 0:
                    a.append(-1)
            for i in range(0, 50):
                max = 0
                count = 0
                for j in range(0, 50):
                    if j == 0 and arr[i][j] == 0:
                        count = count + 1;
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i][j - 1] == 0:
                        count = count + 1
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i][j - 1] == 1:
                        count = 1
                        if (count > max):
                            max = count
                a.append(max)
            #print('3', end='')
            for i in range(0, 50):
                max = 0
                count = 0
                for j in range(0, 50):
                    if j == 0 and arr[j][i] == 0:
                        count = count + 1;
                        if (count > max):
                            max = count
                    elif arr[j][i] == 0 and arr[j - 1][i] == 0:
                        count = count + 1
                        if (count > max):
                            max = count
                    elif arr[j][i] == 0 and arr[j - 1][i] == 1:
                        count = 1
                        if (count > max):
                            max = count
                a.append(max)
            #print('4', end='')
            for k in range(0, 49, 1):
                i = k
                j = 0
                max = 0
                count = 0
                while (i < 50):
                    if j == 0 and arr[i][j] == 0:
                        count = count + 1;
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i - 1][j - 1] == 0:
                        count = count + 1
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i - 1][j - 1] == 1:
                        count = 1
                        if (count > max):
                            max = count
                    i = i + 1
                    j = j + 1
                a.append(max)
            #print('5', end='')
            for k in range(1, 49, 1):
                j = k
                i = 0
                max = 0
                count = 0
                while (j < 50):
                    if i == 0 and arr[i][j] == 0:
                        count = count + 1;
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i - 1][j - 1] == 0:
                        count = count + 1
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i - 1][j - 1] == 1:
                        count = 1
                        if (count > max):
                            max = count
                    i = i + 1
                    j = j + 1
                a.append(max)
            #print('6', end='')
            for k in range(0, 49, 1):
                i = k
                j = 49
                max = 0
                count = 0
                while (i < 50):
                    if j == 49 and arr[i][j] == 0:
                        count = count + 1;
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i - 1][j + 1] == 0:
                        count = count + 1
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i - 1][j + 1] == 1:
                        count = 1
                        if (count > max):
                            max = count
                    i = i + 1
                    j = j - 1
                a.append(max)
            #print('7', end='')
            for k in range(48, 0, -1):
                j = k
                i = 0
                max = 0
                count = 0
                while (j >= 0):
                    if i == 0 and arr[i][j] == 0:
                        count = count + 1;
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i - 1][j + 1] == 0:
                        count = count + 1
                        if (count > max):
                            max = count
                    elif arr[i][j] == 0 and arr[i - 1][j + 1] == 1:
                        count = 1
                        if (count > max):
                            max = count
                    i = i + 1
                    j = j - 1
                a.append(max)
            thefile.write(",".join(map(lambda x: str(x), a)))
            thefile.write("," + str(X) + "\n");
            data.append(a)
            # print(len(a))
        thefile.close()
    except Exception as e:
        print(e)

print()
for j in range(len(data)):
    print('For sample '+str(j))
    for i in range(10):
        clf=joblib.load('Model_final_'+str(i)+'.pkl')
        #print(i,clf.predict(data))
        res=clf.predict(data)[j]
        if res==1:
            print(i,end=' ')
    print()