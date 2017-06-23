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

#fname = 'data_csv/projects/projects.csv.ab'
fname = 'data/opendata_projects000.gz'

projects = ProjectsData(fname)
projects.get_data()
# projects.sample(frac = .5)

# Unpickle the fitted pipeline and the model coefs
filename_model = 'models/pipe_logit_lasso_price.pkl'
pipe_price = pickle.load(open(filename_model, 'rb'))
filename_model = 'models/pipe_logit_lasso.pkl'
pipe = pickle.load(open(filename_model, 'rb'))

# Compute Optimal Price
from scipy import optimize
row = projects.df.iloc[[0]]
def expected_payoff(price, row, pipe):
    row.total_price_excluding_optional_support = price
    p_hat = pipe_price.predict_proba(row)[0][1]
    expected = p_hat * price
    return(-expected)

row = projects.df.iloc[[0]]
res = optimize.minimize_scalar(expected_payoff, args = (row, pipe), options = {'disp':False})
print(res.x)
price_opt = res.x

# Create OddsCheck price column
df = projects.df
df['p_hat'] = pipe.predict_proba(df)[:, 1]
df['price_oc'] = df.total_price_excluding_optional_support
df.price_oc[df.price_oc > price_opt] = price_opt
df_oc_price = df.copy()
df_oc_price['total_price_excluding_optional_support'] = df.price_oc
df['p_hat_oc'] = pipe.predict_proba(df_oc_price)[:, 1]
df['exp_payoff'] = df['total_price_excluding_optional_support'] * df['p_hat']
df['payoff'] = df['funded'] * df['total_price_excluding_optional_support']
df['exp_payoff_oc'] = df['price_oc'] * df['p_hat_oc']

# Look at the projects that get an OddsCheck recommended price
total_payoff = df.payoff[df['total_price_excluding_optional_support'] > price_opt].sum()
total_exp_payoff = df.exp_payoff[df['total_price_excluding_optional_support'] > price_opt].sum()
total_exp_payoff_oc = df.exp_payoff_oc[df['total_price_excluding_optional_support'] > price_opt].sum()

print()
print('For projects that get an OddsCheck reccommendation')k:w
print('total expected payoff')
print(total_exp_payoff)
print('total expected payoff under OddsCheck')
print(total_exp_payoff_oc)
print()
print('Percent gain over expected payoff')
print(total_exp_payoff_oc/total_exp_payoff)
print('Total $ gained over expected')
print(total_exp_payoff_oc - total_exp_payoff)
