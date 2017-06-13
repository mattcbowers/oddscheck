
# coding: utf-8

# # Logistic Regression Model for Project Completion

# In[21]:

import numpy as np
import pandas as pd
from sklearn import preprocessing


# In[22]:

# import a data frame
projects = pd.read_csv('data_csv/projects/projects.csv.aa', escapechar='\\', names=['_projectid', '_teacher_acctid', '_schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject', 'primary_focus_area' ,'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'vendor_shipping_charges', 'sales_tax', 'payment_processing_charges', 'fulfillment_labor_materials', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'funding_status', 'date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration'])
projects.shape


# In[23]:

# keep only columns that have less than 5% missing
def get_data(fname):
    df = pd.read_csv(fname, escapechar='\\', names=['_projectid', '_teacher_acctid', '_schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject', 'primary_focus_area' ,'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'vendor_shipping_charges', 'sales_tax', 'payment_processing_charges', 'fulfillment_labor_materials', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'funding_status', 'date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration'])
    na_counts = df.isnull().sum()
    na_tolerance = 0.05
    cols_to_keep = list(na_counts.loc[na_counts < na_tolerance * na_counts.shape[0]].index)
    df = df[cols_to_keep]
    df['funded'] = df['funding_status'] == 'completed'
    return(df.dropna())


df_train = get_data('data_csv/projects/projects.csv.aa')
df_test = get_data('data_csv/projects/projects.csv.ab')


# ## Just get a logistic model working

# In[45]:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

Y_train = df_train['funded'] 
X_train = df_train[['total_price_excluding_optional_support', 'students_reached']]
Y_test = df_test['funded'] 
X_test = df_test[['total_price_excluding_optional_support', 'students_reached']]


# In[46]:

X_train_scale = scale(X_train)
X_test_scale = scale(X_test)
log=LogisticRegression(penalty='l2',C=.01)
log.fit(X_train_scale,Y_train)
# Checking the model's accuracy
accuracy_score(Y_test,log.predict(X_test_scale))


# ## Logistic Regression Pipeline

# In[47]:

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale

#Y_train = df['funded'] 
#X_train = df[['total_price_excluding_optional_support', 'students_reached']]


# In[50]:

# scale my data
prep = preprocessing.StandardScaler()
# encode categorical variables
#enc = preprocessing.OneHotEncoder()
# Using a LogisticRegression with l1 penalty (Lasso)
Classifier = LogisticRegression(penalty="l1")
# choose metric
score = make_scorer(accuracy_score)
# C parameter space
Cs = 10 ** np.array(range(-2, 4))

# Setting up Pipeline
pipe = Pipeline(steps=[('preprocess', prep), ('logistic', Classifier)])

# Hyperparameter space parameters to be evaluated by a grid search
#n_components = range(60,70,5) # number of Principle components retained
#Cs = np.logspace(0, 4, 5) # Strength of L1 regularization
#n_components = [65,66]
#Cs =[1]
# Tuning hyper-parameters using Grid Search and 5 fold Cross-Validation
print("# Tuning hyper-parameters for %s" % score)
print()
Logistic = GridSearchCV(pipe,
                     dict(logistic__C = Cs),
                     cv = 5,
                     scoring = score)
# Training best model using results from Grid Search
Logistic.fit(X_train_scale, Y_train)
print("Best parameters set found on development set:")
print()
print(Logistic.best_params_)
print()
print("Grid scores on development set:")
print()


# In[59]:

means = Logistic.cv_results_['mean_test_score']
stds = Logistic.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, Logistic.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


# In[ ]:

[8:24] 
```
```

[8:24] 
 ```print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
# Testing trained algorithm on Test Data
y_true, y_pred = Y_test, Logistic.predict(X_test)
# printing Classification report
print(classification_report(y_true, y_pred,target_names=class_names))
print()```

