import statsmodels.api as sm
import pandas as pd
from bld.project_paths import project_paths_join as ppj
import pickle
from datetime import datetime

file = pd.read_csv(ppj("OUT_DATA", "data.csv"), sep = "," )


# Program begins from here
start_time = datetime.now()


# DATA ANALYSIS

# Adding coefficient to the file
file['constant'] = 1

# Basic Mincer Model
X = file[['constant', 'schooling_years', 'experience', 'experience_squared']]
y = file['log_wages']
model_1 = sm.OLS(y, X).fit()
predictions_mincer = model_1.predict(X)
basic = model_1.summary()

with open(ppj("OUT_ANALYSIS", "basic.pickle"), "wb") as out_file:
    pickle.dump(basic, out_file)

# Basic Mincer plus Cognitive
X = file[['constant', 'schooling_years', 'experience', 'experience_squared', 'fluency', 'symbol']]
y = file['log_wages']
model_2 = sm.OLS(y, X).fit()
predictions_cog = model_2.predict(X)
cognitive = model_2.summary()

with open(ppj("OUT_ANALYSIS", "cognitive.pickle"), "wb") as out_file:
    pickle.dump(cognitive, out_file)

X = file[
    ['constant', 'schooling_years', 'experience', 'experience_squared', 'openness', 'conscientiousness', 'extraversion'
        , 'agreeableness', 'neuroticism']]
y = file['log_wages']
model_3 = sm.OLS(y, X).fit()
predictions_noncog = model_3.predict(X)
non_cognitive = model_3.summary()

with open(ppj("OUT_ANALYSIS", "non_cognitive.pickle"), "wb") as out_file:
    pickle.dump(non_cognitive, out_file)

# Basic Mincer plus both

X = file[['constant', 'schooling_years', 'experience', 'experience_squared', 'fluency', 'symbol', 'openness',
          'conscientiousness', 'extraversion'
    , 'agreeableness', 'neuroticism']]
y = file['log_wages']
model_4 = sm.OLS(y, X).fit()
predictions_all = model_4.predict(X)
both = model_4.summary()

with open(ppj("OUT_ANALYSIS", "both.pickle"), "wb") as out_file:
    pickle.dump(both, out_file)

# Regression for Occupation Groups

print('Regression results for different Occupational groups (1,2,3,4)')

for groups in file.occupation.unique():
    tempfile = file[file.occupation == groups]
    X = tempfile[['constant', 'schooling_years', 'experience', 'experience_squared', 'fluency', 'symbol', 'openness',
                  'conscientiousness', 'extraversion'
        , 'agreeableness', 'neuroticism']]
    y = tempfile['log_wages']

    model_5 = sm.OLS(y, X).fit()
    Occupational_results = model_5.summary()
    with open(ppj("OUT_ANALYSIS", "Occupational_results.pickle"), "wb") as out_file:
        pickle.dump(Occupational_results, out_file)
