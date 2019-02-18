import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj

# Program begins from here
start_time = datetime.now()

# Loading all relevant stata files

vp = pd.read_stata(ppj("IN_DATA","vpenglish.dta"))
other = pd.read_stata(ppj("IN_DATA","vpgenenglish.dta"))
job = pd.read_stata(ppj("IN_DATA","vpequiv.dta"))
cog = pd.read_stata(ppj("IN_DATA","cognitenglish.dta"))

# Merging all the files and locating the relevant columns
file = pd.merge(pd.merge(pd.merge(cog, vp, on='persnr'), other, on='persnr'), job, on='persnr')
file = file.loc[:,
       ['persnr', 'vp62', 'vbilzeit', 'egp88_05', 'f96t90g', 'f99z90r', 'vp12501', 'vp12502', 'vp12503', 'vp12504',
        'vp12505', 'vp12506', 'vp12507', 'vp12508', 'vp12509', 'vp12510', 'vp12511', 'vp12512', 'vp12513',
        'vp12514', 'vp12515', 'd1110905', 'd1110105', 'ijob105']]

# Converting the Categorical type variables to Numeric type
cat_columns = file.select_dtypes(['category']).columns
file[cat_columns] = file[cat_columns].apply(lambda x: x.cat.codes)
file[cat_columns] = file[cat_columns].astype('int64')

# Renaming Relevant Variables
file.rename(
    columns={"vp62": "weekly_hours", "ijob105": "wages", "d1110905": "schooling_years", "egp88_05": "occupation",
             "d1110105": "age", }, inplace=True)