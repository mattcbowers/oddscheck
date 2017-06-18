# Estimate the impact of using OddsCheck
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing 
import sys
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.pipeline import Pipeline
from source.utils import ProjectsData
import warnings
warnings.filterwarnings('ignore')
import random
random.seed(55)

fname = 'data_csv/projects/projects.csv.ab'
# fname = 'data/opendata_projects000.gz'

projects = ProjectsData(fname)
projects.get_data()
# projects.sample(frac = .5)

# Unpickle the fitted pipeline and the model coefs
filename_model = 'models/pipe_logit_lasso.pkl'
pipe = pickle.load(open(filename_model, 'rb'))

# Compute Optimal Price
row = projects.df.iloc[[0]]
#prices = np.linspace(1, 10001)
prices = np.arange(1, 100)
pipe.predict_proba(row)

from scipy import optimize
def g(x, y):
    return((x-y)**2)

y = 15
res = optimize.minimize_scalar(g, args = (y), options = {'disp':False})
print(res.x)

def expected_payoff(price, row, pipe):
    row.total_price_excluding_optional_support = price
    p_hat = pipe.predict_proba(row)[0][1]
    expected = p_hat * price
    return(-expected)

row = projects.df.iloc[[0]]
res = optimize.minimize_scalar(expected_payoff, args = (row, pipe), options = {'disp':False})
print(res.x)
