import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing 
from sklearn import metrics
import sys
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.pipeline import Pipeline
from scipy import optimize


import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
def generate_output(query):
    cost = float(query)
    # create data frame from query
    new_data = pd.DataFrame({
        'total_price_excluding_optional_support': [cost],
        'students_reached': [15]
    })
    filename_model = 'flaskexample/models/mod_logit_simple.pkl'
    filename_scaler = 'flaskexample/models/mod_logit_simple_scaler.pkl'
    mod_logit = pickle.load(open(filename_model, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))
    prob = mod_logit.predict_proba(scaler.transform(new_data))[0][1]
    prob_str = str(int(round(100 * prob))) + '%'
    out_str = 'Proposed cost: $' + query + ' Funding probability: ' + prob_str
    return(out_str)

def generate_maybe(query):
    if (float(query) > 1233):
        out_str = 'Consider requesting: $1233, Funding probability: 63%' 
    else:
        out_str = ''
    return(out_str)

def get_probability(resource, grade, prefix, state, poverty, query):
    price = float(query)
    # create data frame from inputs
    # Note that 0 coef fields still need realistic values, but don't affect p
    new_data = pd.DataFrame({
        'total_price_excluding_optional_support': price,
        'students_reached': 0,
        'school_state': state,
        'school_charter': 'f',
        'school_magnet': 'f',
        'school_year_round': 'f',
        'school_nlns': 'f',
        'school_kipp': 'f',
        'school_charter_ready_promise': 'f',
        'teacher_prefix': prefix,
        'teacher_teach_for_america': 'f',
        'teacher_ny_teaching_fellow': 'f',
        'primary_focus_area': 'Applied Learning',
        'resource_type': resource,
        'poverty_level': poverty,
        'grade_level': grade
    }, index = [0])
    filename_model = 'flaskexample/models/pipe_logit_lasso.pkl'
    pipe = pickle.load(open(filename_model, 'rb'))
    prob = pipe.predict_proba(new_data)[0, 1]
    prob_str = str(int(round(100 * prob))) + '%'
    # Add optimal price recommendation
    def expected_payoff(price, row, pipe):
        row.total_price_excluding_optional_support = price
        p_hat = pipe.predict_proba(row)[0, 1]
        expected = p_hat * price
        return(-expected)
    opt = optimize.minimize_scalar(expected_payoff, args = (new_data, pipe), options = {'disp':False})
    price_opt = opt.x
    opt_data = new_data.copy()
    opt_data.total_price_excluding_optional_support = price_opt
    prob_opt = pipe.predict_proba(opt_data)[0, 1]
    prob_opt_str = str(int(round(100 * prob_opt))) + '%'
    out_str1 = 'Proposed price: $' + query + ' Funding probability: ' + prob_str
    out_str2 = ''
    if price > price_opt:
        out_str2 = 'Recommended price: $' + str(int(round(price_opt))) + ' Funding probability: ' + prob_opt_str
    return(out_str1, out_str2)
