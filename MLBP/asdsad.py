import numpy as np

a = [1,2,3]
a = np.array(a)
a = np.reshape(a,(3))
b = [[1,2],[3,4],[5,6],[5,5],[1,3]]
#a = np.array(a)
b = np.array(b)
print(b[a])



import random
for i in range(200):
    print(str(i))