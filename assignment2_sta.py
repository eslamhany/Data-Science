import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

##STEP 1
die = pd.DataFrame(np.random.uniform(low=1,high=7,size=(1000,1)).astype('uint8'))

##STEP 2
die.hist(bins=100)
plt.show()

##STEP 3
def dies(x):
    i_n = pd.DataFrame(data=None)
    for i in range(x):
        i = pd.DataFrame(np.random.uniform(low=1,high=7,size=(1000,1)).astype('uint8'))
        i_n = pd.concat([i_n,i],axis=1,ignore_index=True)
    return i_n


i_n = dies(2)

##STEP 4
average_i_n = i_n.mean(axis=1)
print("Average for TWO Dies = ",average_i_n)

##STEP 5
average_i_n.hist(bins=100)
plt.show()

##STEP 6
die_2_mean = average_i_n.mean()
die_2_var = average_i_n.var()

print("Mean of TWO Dies = ",die_2_mean)
print("Variance of TWO Dies = ",die_2_var)

##STEP 7
i_n10 = dies(10)

##STEP 8
average_i_n10 = i_n10.mean(axis=1)
print("Average of TEN Dies = ",average_i_n10)

##STEP 9
average_i_n10.hist(bins=100)
plt.show()

##STEP 10
die_10_mean = average_i_n10.mean()
die_10_var = average_i_n10.var()

print("Mean of TEN Dies = ",die_10_mean)
print("Variance of Ten Dies = ",die_10_var)

