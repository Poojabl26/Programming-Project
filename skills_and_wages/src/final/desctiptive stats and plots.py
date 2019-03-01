import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import pylab as pl


file = pd.read_csv(ppj("OUT_DATA", "data.csv"), sep = "," )

#Descriptive tables

print(file.describe())
print(file.groupby('occupation').describe())

#descriptive plots

#Correlation Heatmap
Var_Corr = file.corr()
sns.set_style(style = 'white')
cmap = sns.diverging_palette(250, 10, as_cmap=True)
heatmp = sns.heatmap(Var_Corr,cmap=cmap, linewidths=1)
plt.savefig(ppj("OUT_figures",'heatmap.png'))
plt.clf()

#Bar Chart Showing Frequency for Occupation
sns.set(style="darkgrid")
bar = sns.countplot(x="occupation", data=file)
plt.savefig(ppj("OUT_figures",'occupation_count.png'))
plt.clf()
