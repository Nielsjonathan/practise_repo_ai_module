autoreload
%autoreload 2
%matplotlib inline

from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display

from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

#Loading the Dataset

PATH = 'data/Boston Housing Dataset/'
df_raw_train = pd.read_csv(f'{PATH}train.csv',low_memory = False)
df_raw_test = pd.read_csv(f'{PATH}test.csv',low_memory = False)
