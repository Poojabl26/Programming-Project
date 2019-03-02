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

# The function returns the information gain based on the positive and negative side split
def get_information_gain(ppos=335, pneg=340, npos=0, nneg=8):
    total = float(ppos + pneg + npos + nneg)
    p_total = float(ppos + pneg)
    n_total = float(npos + nneg)
    information_gain = entropy((ppos + npos) / total, (pneg + nneg) / total)
    if p_total > 0:
        information_gain -= p_total / total * entropy(ppos / p_total, pneg / p_total)
    if n_total > 0:
        information_gain -= n_total / total * entropy(npos / n_total, nneg / n_total)
    return information_gain

# This calculates the entropy
def entropy(p, n):
    if n == 0:
        return p * math.log(1.0 / p, 2)
    elif p == 0:
        return n * math.log(1.0 / n, 2)
    return p * math.log(1.0 / p, 2) + n * math.log(1.0 / n, 2)

# This one calculates the total number of positive and negative outputs
def number_of_positives(dflocal):
    rowaxes, columnaxes = dflocal.axes
    number_of_positives = 0
    number_of_negatives = 0
    for i in range(len(rowaxes)):
        if (dflocal.iat[i, -1] == 1.0):
            number_of_positives += 1
        else:
            number_of_negatives += 1
    return number_of_positives, number_of_negatives

# Determines the tree recursively by finding the best node heuristically
def Create_tree_TDIDT(node_list, dfa, current_node_id, tree_depth):
    current_node = node_list[current_node_id]

    rowaxes, columnaxes = dfa.axes
    pplus, pminus = number_of_positives(dfa)

    network_information_gain = 0
    final_mean = 0
    node_attribute = 0
    final_cutpoint = 0

    for current_column in range(len(columnaxes) - 1):
        df_temp = dfa.sort_values(by=[columnaxes[current_column]])
        sorted_array = df_temp[:][columnaxes[current_column]]
        result = df_temp[:][columnaxes[-1]]
        # print(result)
        pinnerplus = 0
        pinnerminus = 0
        max_information_gain = 0
        prev_out = 2
        for i in range(len(rowaxes)):
            if (df_temp.iat[i, -1] == 1.0):
                pinnerplus += 1
                information_gain = get_information_gain(ppos=pinnerplus, pneg=pinnerminus,
                                                        npos=(pplus - pinnerplus), nneg=(pminus - pinnerminus))
                if (information_gain > max_information_gain):
                    max_information_gain = information_gain
                    potential_cutpoint = i
                    if i > 0:
                        potential_mean = (df_temp.iat[i, current_column] +
                                          df_temp.iat[i - 1, current_column]) / 2;
                    else:
                        potential_mean = df_temp.iat[i, current_column];
            else:
                pinnerminus += 1
        if (max_information_gain > network_information_gain):
            network_information_gain = max_information_gain
            node_attribute = current_column
            final_mean = potential_mean
            final_cutpoint = potential_cutpoint
    #    print('network_information_gain',network_information_gain)
    #    print('node_attribute',node_attribute)
    #    print('final_mean',final_mean)
    #    print('final_cutpoint',final_cutpoint)
    #    print(columnaxes[node_attribute])
    #    print('-----------------------------')

    # Updating the current array
    current_node.threshold = final_mean
    current_node.pplus = pplus
    current_node.pminus = pminus
    current_node.label = columnaxes[node_attribute]
    # The array is sorted and split
    df_temp = dfa.sort_values(by=[columnaxes[node_attribute]])
    df1 = df_temp.iloc[:final_cutpoint, :]
    df2 = df_temp.iloc[final_cutpoint:, :]

    if pplus == 0 or pminus == 0 or final_cutpoint == 0 or tree_depth >= 3:
        current_node.is_leaf = True
        current_node.outcome = (pplus > pminus)
        return
    else:
        current_node.is_leaf = False

    left_node = TDIDTNode(current_node_id)
    right_node = TDIDTNode(current_node_id)

    current_node.left_child_id = len(node_list)
    current_node.right_child_id = len(node_list) + 1

    # only needed to fullfill exercise requirements
    left_node.identifier = current_node.left_child_id
    right_node.identifier = current_node.right_child_id
    left_node.parent_test_outcome = "yes"
    right_node.parent_test_outcome = "no"

    node_list.append(left_node)
    node_list.append(right_node)
    node_list[current_node.left_child_id].identifier = current_node.left_child_id;
    node_list[current_node.right_child_id].identifier = current_node.right_child_id;
    #    node_list[current_node.left_child_id].is_Left = True
    Create_tree_TDIDT(node_list, df1, current_node.left_child_id, tree_depth + 1)
    Create_tree_TDIDT(node_list, df2, current_node.right_child_id, tree_depth + 1)

    return df_temp

# Parses through the decision tree to find outcome.
def classify(row, dftest, node_list):
    current_node = node_list[0]

    while not current_node.is_leaf:
        if (dftest.iat[row, dftest.columns.get_loc(str(current_node.label))] < current_node.threshold):
            current_node = node_list[current_node.left_child_id]
        else:
            current_node = node_list[current_node.right_child_id]
    return current_node.outcome

# Compares the predicted output to actual output and prints the likelihood
def test_data_output(dftest, node_list):
    rowaxes, columnaxes = dftest.axes
    number_of_matches = 0;
    for row in range(len(rowaxes)):
        predict_op = classify(row, dftest, node_list)
        if (dftest.iat[row, -1] == predict_op):
            number_of_matches += 1
    print('Out of', len(rowaxes), 'tests run, ', number_of_matches,
          'matched the result which is at %', number_of_matches / len(rowaxes))

# To write the node into dot file
def Export_tree_node(node_list, index):
    if index == None:
        return
    Update_to_dot_file(node_list, node_list[index])
    Export_tree_node(node_list, node_list[index].left_child_id)
    Export_tree_node(node_list, node_list[index].right_child_id)


# To write the node into dot file
def Update_to_dot_file(node_list, node):
    # create node
    if (node.is_leaf and (node.outcome)):
        node_description = str(node.identifier) + " [ label=\"" + node.label + "[" + str(node.pplus) + " " + str(
            node.pminus) + "]" + "\" , fillcolor=\"#99ff99\"] ;\n"
    elif (node.is_leaf and (node.outcome == False)):
        node_description = str(node.identifier) + " [ label=\"" + node.label + "[" + str(node.pplus) + " " + str(
            node.pminus) + "]" + "\" , fillcolor=\"#ff9999\"] ;\n"
    else:
        node_description = str(node.identifier) + " [ label=\"" + node.label + "[" + str(node.pplus) + " " + str(
            node.pminus) + "]" + "\" , fillcolor=\"#ffffff\"] ;\n"

    fo.write(node_description)

    if (node.parent_id != -1):
        # create relation
        condition = node.identifier % 2
        if (condition):
            node_relation = str(node.parent_id) + "->" + str(
                node.identifier) + " [labeldistance=2.5, labelangle=45, headlabel=\"<" + str(
                node_list[node.parent_id].threshold) + "\"] ;\n"
        else:
            node_relation = str(node.parent_id) + "->" + str(
                node.identifier) + " [labeldistance=2.5, labelangle=-45, headlabel=\">" + str(
                node_list[node.parent_id].threshold) + "\"] ;\n"
        fo.write(node_relation)
    return

# run TDIDT
node_list = [TDIDTNode()]
k = Create_tree_TDIDT(node_list,train,0,0)

# For exporting the decision tree
fo=open(ppj("OUT_ANALYSIS", "decision_tree.dot"),"w")
print("Name of the dot file: ",fo.name)
fo.write("digraph Tree {\nnode [shape=box, style=\"filled\", color=\"black\"] ;\n")
Export_tree_node(node_list, 0)

fo.write("}")
fo.close()

#use http://www.webgraphviz.com/ to get the png

# print all nodes created by TDIDT
print('The following are the nodes created in the decision tree')
for node in node_list:
    print(node)

# Data set validation
df_validation = test
test_data_output(test, node_list)

print("pooja")
