import pandas as pd
import scipy.stats as st
import math

data_set = pd.DataFrame.from_csv("D:/NU/Statistics/Assignment01Statistics/EUstock.csv")
dax = data_set['DAX']

f_part, s_part = dax[:len(dax)//2], dax[len(dax)//2:]

f_part_mean = f_part.mean()
f_part_var = f_part.var()
print("Mean for first half = ",f_part_mean)
print("Variance for first half = ",f_part_var)

s_part_mean = s_part.mean()
s_part_var = s_part.var()
print("Mean for second half = ",s_part_mean)
print("Variance for second half = ",s_part_var)

q = s_part_mean - f_part_mean
print("Quantity q = ",q)

q_var = s_part_var - f_part_var
print("Variance of q = ",q_var)

a = 0.05
n = len(dax)
x = dax.mean()
s = dax.std()
m = 2500

t = st.t.ppf((1-a/2),n-1)
print(t)

sq = math.sqrt(n)
t_cal = (x-m)/(s/sq)
print(t_cal)

h_test = st.stats.ttest_1samp(a=dax,popmean=m)
print(h_test)

#p>.05, therefore we fail to reject the null hypothesis.