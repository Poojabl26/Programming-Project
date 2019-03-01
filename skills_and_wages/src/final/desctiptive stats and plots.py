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

