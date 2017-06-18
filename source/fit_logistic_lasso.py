# Fit a Lasso Logistic Regression using all 16 features
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing 
from sklearn import metrics
import sys
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split
from source.utils import ProjectsData
import warnings
warnings.filterwarnings('ignore')
import random
random.seed(55)

# fname = 'data_csv/projects/projects.csv.ab'
fname = 'data/opendata_projects000.gz'

projects = ProjectsData(fname)
projects.get_data()
# projects.sample(frac = .5)
projects.train_test_split(train_size = .67)
projects.balance()
projects.X_Y_split(y_col = 'funded')
X_train = projects.X_train
Y_train = projects.Y_train
X_test = projects.X_test
Y_test = projects.Y_test


# Modeling -------------------------------------------------------------
mapper = DataFrameMapper([
    ('total_price_excluding_optional_support', preprocessing.StandardScaler()),
    ('students_reached', preprocessing.StandardScaler()),
    ('school_state', preprocessing.LabelBinarizer()),
    ('school_charter', preprocessing.LabelBinarizer()),
    ('school_magnet', preprocessing.LabelBinarizer()),
    ('school_year_round', preprocessing.LabelBinarizer()),
    ('school_nlns', preprocessing.LabelBinarizer()),
    ('school_kipp', preprocessing.LabelBinarizer()),
    ('school_charter_ready_promise', preprocessing.LabelBinarizer()),
    ('teacher_prefix', preprocessing.LabelBinarizer()),
    ('teacher_teach_for_america', preprocessing.LabelBinarizer()),
    ('teacher_ny_teaching_fellow', preprocessing.LabelBinarizer()),
    ('primary_focus_area', preprocessing.LabelBinarizer()),
    ('resource_type', preprocessing.LabelBinarizer()),
    ('poverty_level', preprocessing.LabelBinarizer()),
    ('grade_level', preprocessing.LabelBinarizer()),
])


# tmp = mapper.fit_transform(df_train.copy())
mapper.fit_transform(X_train.copy())
feature_names = mapper.transformed_names_

# fit logistic with fixed C
# C0 = .001
# C0 = .0005
# C0 = .0004 # adds NY and Literacy from .0003
C0 = .0003 # nice
# C0 = .0001
print('C =  ' + str(C0))
logistic0 = LogisticRegression(penalty='l1', C = C0)
pipe0 = Pipeline(steps = [
  ('mapper', mapper),
  ('logistic0', logistic0)
])
pipe0.fit(X_train, Y_train)

# The best model
print('coefficients of best model')
fitted_coefs = pd.DataFrame({'name':feature_names, 'coef':pipe0.named_steps['logistic0'].coef_[0]}).sort('coef')
print(fitted_coefs[fitted_coefs.coef != 0])

# Testing trained algorithm on Test Data
y_true, y_pred = Y_test, pipe0.predict(X_test)
# printing Classification report
print(metrics.classification_report(y_true, y_pred,target_names= None))
print()

# Pckle the fitted pipeline and the model coefs
filename_model = 'models/pipe_logit_lasso.pkl'
filename_coefs = 'source/data_pkl/coef_lasso.pkl'
pickle.dump(pipe0, open(filename_model, 'wb'))
pickle.dump(fitted_coefs, open(filename_coefs, 'wb'))

# write the fitted coefs to a csv
fitted_coefs.to_csv('source/data_pkl/coef_lasso.csv', header=None, index=None, mode='a')


