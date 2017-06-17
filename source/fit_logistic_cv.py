# Logistic Regression with Cross Validation and L1 parameter selection
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
from sklearn.model_selection import train_test_split
from source.utils import ProjectsData
import random
random.seed(55)

#df_train = get_data('data_csv/projects/projects.csv.aa')
#df_test = get_data('data_csv/projects/projects.csv.ab')

fname = 'data_csv/projects/projects.csv.ab'
# fname = 'data/opendata_projects000.gz'

#df = get_data(fname)
projects = ProjectsData(fname)
projects.get_data()
projects.sample(frac = .5)
projects.train_test_split(train_size = .67)

sys.exit()
# balance the test and train sets
def make_balanced(df):
    num_true = df.funded.sum()
    num_false = df.shape[0] - num_true
    df_false= df[df['funded'] == False]
    df_true = df[df['funded'] == True]
    df_true_sub = df_true.sample(n = num_false)
    df_balance = pd.concat([df_true_sub, df_false])
    return(df_balance)

df_train = make_balanced(df_train_raw)
df_test = make_balanced(df_test_raw)

# Split data into test and training
def df_to_X(df):
    return(df.drop('funded', axis = 1))
def df_to_Y(df):
    return(df['funded'])

X_train = df_to_X(df_train)
Y_train = df_to_Y(df_train)
X_test = df_to_X(df_test)
Y_test = df_to_Y(df_test)

# Make lists of the different kinds of columns
def get_df_types(df, my_type):
    mylist = list(df.select_dtypes(include=[my_type]).columns)
    return(mylist)

cols_numeric = get_df_types(df_train, 'float64')
cols_binary = get_df_types(df_train, 'bool')
cols_str = get_df_types(df_train, 'object')

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

"""
mapper = DataFrameMapper([
    ('total_price_excluding_optional_support', preprocessing.StandardScaler()),
#    ('students_reached', preprocessing.StandardScaler()),
#    ('school_state', preprocessing.LabelBinarizer()),
#    ('school_charter', preprocessing.LabelBinarizer()),
#    ('school_magnet', preprocessing.LabelBinarizer()),
#    ('school_year_round', preprocessing.LabelBinarizer()),
#    ('school_nlns', preprocessing.LabelBinarizer()),
#    ('school_kipp', preprocessing.LabelBinarizer()),
#    ('school_charter_ready_promise', preprocessing.LabelBinarizer()),
#    ('teacher_prefix', preprocessing.LabelBinarizer()),
#    ('teacher_teach_for_america', preprocessing.LabelBinarizer()),
#    ('teacher_ny_teaching_fellow', preprocessing.LabelBinarizer()),
#    ('primary_focus_area', preprocessing.LabelBinarizer()),
    ('resource_type', preprocessing.LabelBinarizer()),
#    ('poverty_level', preprocessing.LabelBinarizer()),
    ('grade_level', preprocessing.LabelBinarizer()),
])
"""

# tmp = mapper.fit_transform(df_train.copy())
mapper.fit_transform(df_train.copy())
feature_names = mapper.transformed_names_

# fit logistic with fixed C
C0 = .001
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

#score = metrics.make_scorer(metrics.accuracy_score)
score = metrics.make_scorer(metrics.f1_score)
Cs =  10 ** np.array(range(-2, 4)) + .001
logistic = LogisticRegression(penalty='l1')
#logistic.fit(mapper.fit_transform(X_train),Y_train)

pipe = Pipeline(steps = [
  ('mapper', mapper),
  ('logistic', logistic)
])

#pipe.fit_transform(X_train, Y_train)
grid = GridSearchCV(
    pipe,
    dict(logistic__C = Cs),
    cv = 5,
    scoring = score
)
grid.fit(X_train, Y_train)
print("Best parameters set found on development set:")
print()
print(grid.best_params_)
print()

# The best model
print('coefficients of best model')
best = grid.best_estimator_.named_steps['logistic']
fitted_coefs = pd.DataFrame({'name':feature_names, 'coef':best.coef_[0]}).sort('coef')
print(fitted_coefs)
#xx=pd.DataFrame({
#    'total_price_excluding_optional_support':1000,
#    'resource_type': 'Trips',
#    'grade_level': 'Grades 9-12'
#})
#print(grid.best_estimator_.predict(xx))

print("Grid scores on development set:")
print()

means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
# Testing trained algorithm on Test Data
y_true, y_pred = Y_test, grid.predict(X_test)
# printing Classification report
print(metrics.classification_report(y_true, y_pred,target_names= None))
print()




sys.exit()
# save the scaler and the model
filename_model = 'models/mod_logit.pkl'
pickle.dump(mod_logit, open(filename_model, 'wb'))
pickle.dump(scaler, open(filename_scaler, 'wb'))

# Load the model
mod_logit_reload = pickle.load(open(filename_model, 'rb'))
scaler_reload = pickle.load(open(filename_scaler, 'rb'))

# Checking the model's accuracy on unseen test data
print("accuracy = ", accuracy_score(Y_test, mod_logit.predict(scaler.transform(X_test))))
print("accuracy = ", accuracy_score(Y_test, mod_logit_reload.predict(scaler_reload.transform(X_test))))

# Scale numeric data and retain the scaler for new data
new_data = pd.DataFrame({
    'total_price_excluding_optional_support': [500], 
    'students_reached': [15]
})
print(mod_logit.predict(scaler.transform(new_data)))
print(mod_logit_reload.predict(scaler_reload.transform(new_data)))
print(mod_logit.predict_proba(scaler.transform(new_data)))

