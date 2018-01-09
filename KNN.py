import numpy as np
import os
import pandas as pd
import scipy.ndimage
from scipy.spatial import distance
import matplotlib.pyplot as plt
import string as str
import collections


np.set_printoptions(threshold=np.nan)

#getting training images array
noise_train = np.empty(shape=[0, 0])
for root, dirs, files in os.walk("D:/NU/Intro to Machine Learning Dr. Saif/Assignments/Machine Learning - Assignment 2/Problem 2 Dataset/Noise Train/"):
    for i, imgname in enumerate(files):
        img_a = scipy.ndimage.imread(os.path.join(root, imgname))
        img_a.reshape(1, 144)
        noise_train = np.append(noise_train, img_a)
noise_train.resize(182, 144)

#getting distance array
dist_arr = np.empty(shape=[0, 0])
for i in range(182):
    for j in range(182):
        dist = distance.euclidean(noise_train[i], noise_train[j])
        dist_arr = np.append(dist_arr, dist)
dist_arr.resize(182, 182)



labels = np.empty(shape=[0,0])
for i in range(26):
    labels = np.append(labels,[i]*7)


#Implemnting Cross Validation and getting errors for K values
final_error = pd.DataFrame(data=None)
for times in range(1, 11):
    indices = np.random.permutation(dist_arr.shape[0])
    training_idx, val_idx = indices[:145], indices[145:]
    training, val = dist_arr[training_idx, :], dist_arr[val_idx, :]
    train = np.array(training)
    validation = np.array(val)

    e_array = np.zeros(shape=(100, 37), dtype=int)
    error = pd.DataFrame(e_array)
    for k in range(100):
        for i in range(37):
            small_dist = np.argpartition(validation[i], k+2)[:k+2]
            c = labels[small_dist[0]]
            a, b = 0, 0
            for j in range(1, len(small_dist)):
                if labels[small_dist[j]] == c:
                    a += 1 #number of photos in same class
                else:
                    b += 1 #number of photos in other classes
            if a > b:
                error[i][k] = 0
            elif a == b:
                even = error[i][k-1]
                error[i][k] = even
            else:
                error[i][k] = 1
    error_avg = error.mean(axis=1) #average for each sample of 10 samples
    final_error = pd.concat([final_error,error_avg],axis=1,ignore_index=True)

final_error_df = final_error.mean(axis=1) #average for all samples for each k

#Plotting errors to determine best K
plt.bar(range(100), final_error_df,  alpha=0.5)
plt.xticks(range(1,101), final_error_df)
xx = np.arange(len(final_error_df))
a=plt.gca()
a.set_xticks(xx)
a.set_xticklabels(range(1,101))
plt.ylabel('error')
plt.savefig('KNN.jpg')
plt.show()


#Getting test images array
noise_test = np.empty(shape=[0, 0])
for root, dirs, files in os.walk("D:/NU/Intro to Machine Learning Dr. Saif/Assignments/Machine Learning - Assignment 2/Problem 2 Dataset/Noise Test/"):
    for i, imgname in enumerate(files):
        img_a = scipy.ndimage.imread(os.path.join(root, imgname))
        img_a.reshape(1, 144)
        noise_test = np.append(noise_test, img_a)
noise_test.resize(52, 144)


#Distance between each image in test array and all images in training array
test_dist_arr = np.empty(shape=[0, 0])
for i in range(52):
    for j in range(182):
        t_dist = distance.euclidean(noise_test[i], noise_train[j])
        test_dist_arr = np.append(test_dist_arr, t_dist)
test_dist_arr.resize(52, 182)


#testing test images
t_e_array = np.zeros(shape=(100, 37), dtype=int)
t_error = pd.DataFrame(t_e_array)
result = []
for i in range(52):
    dist_index = np.argpartition(test_dist_arr[i], 1)[:1]
    CT = int(labels[dist_index])
    group = str.ascii_lowercase[CT]
    result.extend([group])

counter = collections.Counter(result)
data_d = counter.most_common(26)
data = sorted(data_d)

#plotting test result
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
