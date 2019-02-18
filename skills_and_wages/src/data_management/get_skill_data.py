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

# Cleaning Data
file = file.loc[(file['age'] > 20) & (file['age'] <= 60)]
Occ_good = [2, 3, 4, 5, 8, 9]
file = file[file['occupation'].isin(Occ_good)]
file = file.replace([-2, -1, 0], np.nan)
file = file.dropna()
# file = file.reset_index(drop=True)
print(file.isnull().sum())

# Standardising test scores for Cognitive Abilities
file['fluency'] = preprocessing.scale(file['f96t90g'])
file['symbol'] = preprocessing.scale(file['f99z90r'])

# Reversing the scale for personality items
file['revvp12507'] = 8 - file['vp12507']
file['revvp12512'] = 8 - file['vp12512']
file['revvp12503'] = 8 - file['vp12503']
file['revvp12515'] = 8 - file['vp12515']

# Aggragting item variables to get Personlity variables and standardising them
file['openness'] = preprocessing.scale((file.iloc[:, [10, 15, 20]].sum(axis=1)) / 3)
file['conscientiousness'] = preprocessing.scale((file.iloc[:, [7, 22, 17]].sum(axis=1)) / 3)
file['extraversion'] = preprocessing.scale((file.iloc[:, [8, 14, 23]].sum(axis=1)) / 3)
file['agreeableness'] = preprocessing.scale((file.iloc[:, [24, 12, 19]].sum(axis=1)) / 3)
file['neuroticism'] = preprocessing.scale((file.iloc[:, [25, 16, 11]].sum(axis=1)) / 3

# Generating Variable for Years of Experience of an individual
file['experience'] = file['age'] - file['schooling_years'] - 6
file['experience_squared'] = file['experience'] ** 2

# Generating log wages
file['log_wages'] = np.log((file['wages'] / (file['weekly_hours'] * (4.3))))

# Merging and renaming similar occupation categories
file = file.replace({'occupation': {2: 1, 9: 8}})  # 1 Higher managers
file = file.replace({'occupation': {3: 2, 8: 6}})  # 2 Lower managers
file = file.replace({'occupation': {5: 3, 4: 3}})  # 3 Routine workers
file = file.replace({'occupation': {6: 4}})  # 4 Manual Workers