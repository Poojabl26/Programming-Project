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
model_1 = sm.OLS(y, X).fit()
predictions_mincer = model_1.predict(X)
print(model_!.summary())

# Basic Mincer plus Cognitive
X = file[['constant', 'schooling_years', 'experience', 'experience_squared', 'fluency', 'symbol']]
y = file['log_wages']
model_2 = sm.OLS(y, X).fit()
predictions_cog = model_2.predict(X)
print(model_2.summary())

X = file[
    ['constant', 'schooling_years', 'experience', 'experience_squared', 'openness', 'conscientiousness', 'extraversion'
        , 'agreeableness', 'neuroticism']]
y = file['log_wages']
model_3 = sm.OLS(y, X).fit()
predictions_noncog = model_3.predict(X)
cognitive = model_3.summary()

# Basic Mincer plus both

X = file[['constant', 'schooling_years', 'experience', 'experience_squared', 'fluency', 'symbol', 'openness',
          'conscientiousness', 'extraversion'
    , 'agreeableness', 'neuroticism']]
y = file['log_wages']
model_4 = sm.OLS(y, X).fit()
predictions_all = model_4.predict(X)
both = model_$.summary()

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