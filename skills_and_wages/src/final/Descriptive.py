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
plt.savefig(ppj("OUT_FIGURES",'heatmap.png'))
plt.clf()

#Bar Chart Showing Frequency for Occupation
sns.set(style="darkgrid")
bar = sns.countplot(x="occupation", data=file)
plt.savefig(ppj("OUT_FIGURES",'occupation_count.png'))
plt.clf()

#Density plot(All Relevant Independent  Variables)
traits = [file['fluency'], file['symbol'], file['openness'], file['conscientiousness'], file['extraversion'], file['agreeableness'],
          file['neuroticism'], file['schooling_years'], file['experience']]
for i, j in enumerate(traits):
    plt.subplot(3, 3, i+1) ,sns.distplot(j, hist=True, kde=True,
             bins=int(180/4), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.savefig(ppj("OUT_FIGURES",'distplot.png'))
