import statsmodels.api as sm
import pandas as pd
from bld.project_paths import project_paths_join as ppj

file = pd.read_csv(ppj("OUT_DATA", "data.csv"), sep = "," )


