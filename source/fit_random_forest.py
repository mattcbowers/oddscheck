# matt bowers 2017-06-13
# Random Forest to predict funding status
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# import a data frame
def get_data(fname, frac_na_tolerable = 0.05):
    df = pd.read_csv(fname, escapechar='\\', names=['_projectid', '_teacher_acctid', '_schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject', 'primary_focus_area' ,'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'vendor_shipping_charges', 'sales_tax', 'payment_processing_charges', 'fulfillment_labor_materials', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'funding_status', 'date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration'])
    na_counts = df.isnull().sum()
    # keep only columns that have less than 5% missing
    cols_to_keep = list(na_counts.loc[na_counts < frac_na_tolerable * na_counts.shape[0]].index)
    df = df[cols_to_keep]
    df['funded'] = df['funding_status'] == 'completed'
    df.drop('funding_status', inplace = True, axis = 1)
#    df = df[(df['funding_status'] == 'completed') or (df['funding_status'] == 'expired')]
    df.drop('date_posted', inplace = True, axis = 1)
    return(df.dropna())

df_train = get_data('data_csv/projects/projects.csv.aa')
df_test = get_data('data_csv/projects/projects.csv.ab')

# Split data into test and training
X_train = df_train.drop('funded', axis = 1)
Y_train = df_train['funded']
X_test = df_train.drop('funded', axis = 1)
Y_test = df_train['funded']

# Encode categorical variables
le=LabelEncoder()
for col in X_test.columns.values:
   # Encoding only categorical variables
   if X_test[col].dtypes=='object':
       # Using whole data to form an exhaustive list of levels
       data=X_train[col].append(X_test[col])
       le.fit(data.values)
       X_train[col]=le.transform(X_train[col])
       X_test[col]=le.transform(X_test[col])

# Fit Random Forest
clf = RandomForestClassifier()
clf.fit(X_train, Y_train)

# Feature importance
names = X_train.columns.values
print("Features sorted by feature importances")
print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names), reverse=True))
