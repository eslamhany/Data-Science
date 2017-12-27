import os
import scipy.ndimage
import numpy as np
import string as str
import collections
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

#getting training data in an array
train_data = np.empty(shape=[0, 0])
for root, dirs, files in os.walk("C:/Users/sezar/Downloads/Compressed/Assignment 1 Dataset/Train/"):
    for i, imgname in enumerate(files):
        img_a = scipy.ndimage.imread(os.path.join(root, imgname))
        img_a.reshape(1, 144)
        img_a = np.append(img_a, 1)
        train_data = np.append(train_data, img_a)
train_data.resize(182, 145)


def t_function(let):
    t = np.array([0] * 182)
    for root, dirs, files in os.walk("C:/Users/sezar/Downloads/Compressed/Assignment 1 Dataset/Train/"):
        for i, img_name in enumerate(files):
            if img_name[2] is let:
                t[i] = t[i] + 1
            else:
                t[i] = t[i] - 1
    return t

#function to get w value
def w_value(let):
    w = np.array([0] * 145)
    w[0] = 1
    for d in range(120):
        for r in range(182):
            if (np.dot(w,train_data[r]) * t_function(let)[r]) < 0:
                w = w + 0.5 * train_data[r] * t_function(let)[r]
    return w

#getting w values in an array to get it once not to call w_value every time
w_array = np.empty(shape=[0,0])
for i in str.ascii_lowercase:
    w_array = np.append(w_array,w_value(i))
w_array.resize(26,145)

#getting test data in an array
test_data = np.empty(shape=[0,0])
for test_root, dir, tests in os.walk("C:/Users/sezar/Downloads/Compressed/Assignment 1 Dataset/Test/"):
    for q, test_me in enumerate(tests):
            img_b = scipy.ndimage.imread(os.path.join(test_root, test_me))
            img_b.reshape(1, 144)
            img_b = np.append(img_b, 1)
            test_data = np.append(test_data, img_b)
test_data.resize(52,145)

#testing test data
result = []
for i in range(52):
    z = 0
    for j in range(26):
        x = np.dot(w_array[j],test_data[i])
        enu = j
        if x > z:
            z = x
            en = enu
            group = str.ascii_lowercase[en]
    result.extend([group])
counter = collections.Counter(result)
data = counter.most_common(26)

#drawing the fogue
labels, y = zip(*data)
x = np.arange(len(labels))
width = 1
fig = plt.figure()
ax=fig.gca()
plt.bar(x, y, align='edge', width=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yticks((y))
plt.savefig('test.jpg')
plt.show()