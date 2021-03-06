import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import pylab as pl
from bld.project_paths import project_paths_join as ppj
import matplotlib.image as mpimg



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

plt.clf()
#Importing the image generated using dot file and exporting it to out figures folder
img=mpimg.imread(ppj("IN_DATA", 'decisiontree.png'))
imgplot = plt.imshow(img)
plt.axis('off')
plt.savefig(ppj("OUT_FIGURES", "tree"))


