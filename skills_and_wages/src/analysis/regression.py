import statsmodels.api as sm
import pandas as pd
from bld.project_paths import project_paths_join as ppj
import pickle
from datetime import datetime
import numpy as np
from random import choice
from numpy import array, dot, random, asfarray
from sklearn import tree
import graphviz as gv
import re
import math
import copy
import csv
import os.path
from sklearn.datasets import load_iris

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

file['wage_binary'] = 0

file.loc[file['log_wages'] > file.log_wages.median(), "wage_binary"] = 1.0
print(file.wage_binary.value_counts())



file = file.loc[:, ['openness', 'conscientiousness', 'extraversion',
       'agreeableness', 'schooling_years', 'fluency', 'symbol', 'neuroticism',
       'experience', 'experience_squared',
       'wage_binary']]


msk = np.random.rand(len(file)) < 0.8

train = file[msk]

test = file[~msk]

#This is a node for building a tree.


class TDIDTNode:

    def __init__(self, parent_id=-1, left_child_id=None, right_child_id=None):
        self.parent_id = parent_id
        self.is_Left = False
        # self.direction      = direction
        self.left_child_id = left_child_id
        self.right_child_id = right_child_id
        self.is_leaf = False
        self.outcome = None
        # only needed to fullfill exercise requirements
        self.identifier = 0
        self.parent_test_outcome = None
        self.pplus = None
        self.pminus = None
        self.label = None
        self.threshold = None

    def setLeftChild(self, id):
        self.left_child_id = id

    def setRightChild(self, id):
        self.right_child_id = id

    def setpplus(self, id):
        self.pplus = id

    def setpminus(self, id):
        self.pminus = id

    def setthreshold(self, id):
        self.threshold = id

    def setlabel(self, id):
        self.label = id

    def setdirection(self, id):
        self.direction = direction

    def setidentifier(self, id):
        self.identifier = id

    def setis_Left(self, id):
        self.is_Left = id

    def __str__(self):
        return "{} {} {} {} ".format(self.label, self.threshold, self.pplus, self.pminus)
