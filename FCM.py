import numpy as np
import os
import scipy.ndimage
from scipy.spatial import distance
import matplotlib.pyplot as plt
import collections
import math


#np.set_printoptions(threshold=np.nan)

train = np.empty(shape=[0, 0])
for root, dirs, files in os.walk("D:/NU/Intro to Machine Learning Dr. Saif/Assignments/Machine Learning - Assignment 3/Assignment 3 Dataset/"):
    for i, imgname in enumerate(files):
        img = scipy.ndimage.imread(os.path.join(root, imgname))
        img.reshape(1, 144)
        train = np.append(train, img)
train.resize(182, 144)


val_dist = np.empty(shape=[0, 0])
for i in range(182):
    for j in range(182):
        dist_arr = distance.euclidean(train[i], train[j])
        val_dist = np.append(val_dist, dist_arr)
val_dist.resize(182, 182)



random_center = val_dist[np.random.randint(val_dist.shape[0], size=1), :]
random_center.resize(182,)
ind_arr = np.empty(shape=[0, 0])
max_dist = random_center.argsort()[-1:][::-1]
ind_arr = np.append(ind_arr, max_dist)

for i in range(25):
    xx_yy = val_dist[max_dist]
    xx_yy.resize(182, )
    max_26_dist = xx_yy.argsort()[-26:][::-1]
    max_26 = sorted(set(max_26_dist).difference(ind_arr))
    max_dist = max_26[0]
    ind_arr = np.append(ind_arr, max_dist)
ind_arr2 = np.empty(shape=[0, 0])
for f in range(0,182,7):
    ind_arr2 = np.append(ind_arr2, f)


max_dist_array = np.empty(shape=[0, 0])
for i in range(26):
    max_dist_array = np.append(max_dist_array, train[int(ind_arr2[i])])
max_dist_array.resize(26, 144)


u = np.zeros(shape=[182, 26])
centers_avr_dist = float("+inf")

while centers_avr_dist >= 0.00001:
    ww = max_dist_array

    point_cluster_dist = np.empty(shape=[0, 0])
    for i in range(182):
        for j in range(26):
            if distance.euclidean(train[i], max_dist_array[j]) == 0:
                train[i] = train[i] + (10 ** (-15))
            p_c_dist = (distance.euclidean(train[i], max_dist_array[j]) ** 2)
            point_cluster_dist = np.append(point_cluster_dist, p_c_dist)
    point_cluster_dist.resize(182, 26)

    q = 1.25
    for i in range(182):
        for k in range(26):
            x = 0
            for j in range(26):
                if math.isnan(point_cluster_dist[i][j]):
                    x = 1
                else:
                    x = x + ((point_cluster_dist[i][k] / point_cluster_dist[i][j]) ** (2 / (q - 1)))
            u[i][k] = 1 / x

    max_dist_array = np.empty(shape=[0, 0])
    for i in range(26):
        s, z = 0, 0
        for j in range(182):
            s = s + ((u[j][i] ** q) * train[j])
            z = z + (u[j][i] ** q)
        d = s / z
        max_dist_array = np.append(max_dist_array, d)
    max_dist_array.resize(26, 144)

    c = 0
    for i in range(26):
        c = c + (distance.euclidean(max_dist_array[i], ww[i]))
        centers_avr_dist = c / 26


result = []
for i in range(182):
    xc = u[i].argsort()[-1:][::-1]
    result.extend(xc+1)


counter = collections.Counter(result)
data_d = counter.most_common(26)
data = sorted(data_d)


lab, y = zip(*data)
x = np.arange(len(lab))
width = 1
fig = plt.figure()
ax=fig.gca()
plt.bar(x, y, align='edge', width=0.8)
ax.set_xticks(x)
ax.set_xticklabels(lab)
ax.set_yticks((y))
plt.savefig('Accuracy.jpg')
plt.show()
