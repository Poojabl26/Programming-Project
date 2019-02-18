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
