import statsmodels.api as sm
import pandas as pd
from bld.project_paths import project_paths_join as ppj

file = pd.read_csv(ppj("OUT_DATA", "data.csv"), sep = "," )


# Program begins from here
start_time = datetime.now()


# DATA ANALYSIS

# Adding coefficient to the file
file['constant'] = 1

# Basic Mincer Model
X = file[['constant', 'schooling_years', 'experience', 'experience_squared']]
y = file['log_wages']
model = sm.OLS(y, X).fit()
predictions_mincer = model.predict(X)
# print(model.summary())

# Basic Mincer plus Cognitive
X = file[['constant', 'schooling_years', 'experience', 'experience_squared', 'fluency', 'symbol']]
y = file['log_wages']
model = sm.OLS(y, X).fit()
predictions_cog = model.predict(X)
# print(model.summary())