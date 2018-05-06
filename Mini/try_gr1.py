import matplotlib.pyplot as plt


'''dig=[i for i in range(1,211,10)]
acc=[23.13,93.32,94,95.44,93.93,94.1,93.8,93.29,96.05,95.33,94.72,94.31,93.96,93.73,93.53,93.37,93.25,93.15,93.11,93.105,93.101]

plt.plot(dig,acc)
plt.xlabel('No of features')
plt.ylabel('Accuracy of one class classifier')
plt.title('PCA')
plt.show()'''


'''dig=[0,1,2,3,4,5,6,7,8,9]
acc=[99.69,97.67,98.36,98.08,98.99,98.42,99.47,99.12,88.78,90.71]


plt.plot(dig,acc,marker='o')
plt.xlabel('Digit')
plt.ylabel('Accuracy of one class classifier')
plt.ylim([0,120])
plt.xlim([0,10])
plt.title('Scan line')
plt.show()
#plt.savefig('Scan line.png')
plt.clf()

dig=[0,1,2,3,4,5,6,7,8,9]
acc=[99.19,97.28,98.04,98.10,97.76,97.93,99.22,98.47,98.45,97.73]

plt.plot(dig,acc,marker='o')
plt.xlabel('Digit')
plt.ylabel('Accuracy of one class classifier')
plt.ylim([0,120])
plt.title('Run length')
#plt.show()
plt.savefig('Run length.png')
plt.clf()

dig=[0,1,2,3,4,5,6,7,8,9]
acc=[97.95,98.27,97.67,97.95,97.79,98.36,97.78,97.82,96.47,96.62]

plt.plot(dig,acc,marker='o')
plt.xlabel('Digit')
plt.ylabel('Accuracy of one class classifier')
plt.ylim([0,120])
plt.title('Zone based')
plt.savefig('Zone based.png')
plt.clf()
plt.clf()'''

dig=[i for i in range(10)]
f1=[137,134,145,141,138,137,137,143,137,142]
f2=[173,155,162,160,157,158,159,156,169,156]
plt.plot(dig,f1,marker='o',label='Incremental PCA')
plt.plot(dig,f2,marker='o',label='Conventional PCA')
plt.xlabel('Digit')
plt.ylabel('No of principal Components required to retain 95% variance of entire dataset')
plt.legend()
plt.savefig('FInally.png')


'''ls=[49, 40, 38, 36, 41, 39, 40, 37, 40]
rev=[]
for i in range(len(ls)-1,-1,-1):
    rev.append(ls[i])
plt.plot([i for i in range(len(rev))],rev,label='No of components after merging',marker='o')
plt.xlabel('No of batches merged')
plt.ylabel('No of features')
plt.ylim([30,80])
plt.legend()
plt.savefig('No of features vs Batches merged for Digit for single pca '+str(3)+'.png')
plt.clf()'''


'''acc1=[75.480844029560487, 84.064177362893815, 85.064177362893815, 85.064177362893815, 85.064177362893815, 85.064177362893815, 85.064177362893815, 85.064177362893815]
acc2=[69.631563593932327, 80.881563593932327, 84.631563593932327, 84.381563593932327, 84.381563593932327, 84.381563593932327, 84.381563593932327, 84.381563593932327, 84.381563593932327, 84.381563593932327]

plt.plot([i for i in range(len(acc1))],acc1,marker='o')
plt.xlabel('No of batches merged')
plt.ylabel('Accuracy')
plt.ylim([0,100])
plt.savefig('final_ac_1.png')
plt.clf()

plt.plot([i for i in range(len(acc2))],acc2,marker='o')
plt.xlabel('No of batches merged')
plt.ylabel('Accuracy')
plt.ylim([0,100])
plt.savefig('final_ac_2.png')'''

