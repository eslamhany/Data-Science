import pandas as pd
import matplotlib.pyplot as plt

data_set = pd.DataFrame.from_csv("D:/NU/Statistics/Assignment01Statistics/EUstock.csv")
ds1 = data_set.loc[:,'DAX']
ds2 = data_set.loc[:,'SMI']

ds1_change = pd.concat([ds1],ignore_index=True).diff().abs()[1:1860]
ds2_change = pd.concat([ds2],ignore_index=True).diff().abs()[1:1860]

def calculate_mean(ds):
    j=0
    for i in range(1,len(ds)+1):
        j+=ds.loc[i]
    ds_mean=j/(len(ds))
    return(ds_mean)

def calculate_var(ds):
    mean_ds=calculate_mean(ds)
    j=0
    for i in range(1,len(ds)+1):
        j+=(ds.loc[i]-mean_ds)**2
    ds_var=j/(len(ds)-1)
    return(ds_var)

ds1ch_mean = float(calculate_mean(ds1_change))
ds2ch_mean = float(calculate_mean(ds2_change))
ds1ch_var = calculate_var(ds1_change)
ds2ch_var = calculate_var(ds2_change)

print('Mean of change in DS1     = ',ds1ch_mean)
print('Variance of change in DS1 = ',ds1ch_var)
print('Mean of change in DS2     = ',ds2ch_mean)
print('Variance of change in DS2 = ',ds2ch_var)

overall_mean = (ds1ch_mean*len(ds1_change) + ds2ch_mean*len(ds2_change))/(len(ds1_change)+len(ds2_change))
print('Overall Mean              = ',overall_mean)


overall_var = ( len(ds1_change)*(ds1ch_var+(ds1ch_mean-overall_mean)**2) + len(ds2_change)*(ds2ch_var+(ds2ch_mean-overall_mean)**2) )/(len(ds1_change)+len(ds2_change))
print('Overall variance          = ',overall_var)

print('DS1 Skewness              = ',ds1_change.skew())
print('DS1 Kurtosis              = ',ds1_change.kurt())

combined_ds = pd.concat([ds1_change,ds2_change],axis=0,ignore_index=True)

plt.figure(1)
plt.boxplot(combined_ds)

plt.figure(2)
plt.hist(combined_ds,bins=100)

plt.show()
