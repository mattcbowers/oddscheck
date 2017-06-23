# utility functions for the python scripts in source/
from sklearn.model_selection import train_test_split
import pandas as pd

class ProjectsData(object):

    def __init__(self, fname):
        self.fname = fname
        
    def get_data(self, frac_na_tolerable = 0.05):
        # all col names
        col_names = ['_projectid', '_teacher_acctid', '_schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject', 'primary_focus_area' ,'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'vendor_shipping_charges', 'sales_tax', 'payment_processing_charges', 'fulfillment_labor_materials', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'funding_status', 'date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration']
        # col names to keep initially
        col_initial_keep = ['funding_status', 'school_state', 'school_metro', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_area', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'total_price_excluding_optional_support', 'students_reached']
        self.df = pd.read_csv(self.fname, escapechar='\\', names = col_names) 
        # convert number students to float
        self.df['date_posted'] = pd.to_datetime(self.df['date_posted'])
        self.df = self.df[self.df['date_posted'] < pd.to_datetime('2016-10-01')]
        print('removed the recent dates with incomplete projects')
        print(self.df.shape)
        self.df.students_reached = self.df.students_reached.astype(float)
        print("Initial size of data")
        print(self.df.shape)
        self.df = self.df[col_initial_keep]
        print("Size of data after removing unnecessary columns apriori")
        print(self.df.shape)
        frac_na = self.df.isnull().mean()
        # keep only columns that have less than 5% missing
        cols_to_keep = list(frac_na.loc[frac_na < frac_na_tolerable].index)
        self.df = self.df[cols_to_keep]
        print("The following columns have > 5% missing and are dropped")
        print(frac_na.loc[frac_na >= frac_na_tolerable].index)
        print(self.df.shape)
        # remove outliers
        price_cap = 10000
        students_reached_cap = 10000
        self.df = self.df[self.df['students_reached'] < students_reached_cap]
        self.df = self.df[self.df['total_price_excluding_optional_support'] < price_cap]
        print("Removing outliers")
        print(self.df.shape)
        # remove rows with NA
        self.df.dropna(inplace = True)
        print("Remove any rows with missing values")
        print(self.df.shape)
        # create binary funded status variable and remove the string version
        self.df['funded'] = self.df['funding_status'] == 'completed'
        self.df.drop('funding_status', axis = 1, inplace = True)

    def sample(self, frac):
        self.df.sample(frac = frac)

    def train_test_split(self, train_size):
        (self.df_train, self.df_test) = train_test_split(self.df, train_size = train_size, random_state = 27)

    def balance(self):
        def downsample_majority(df):
            num_true = df.funded.sum()
            num_false = df.shape[0] - num_true
            df_false= df[df['funded'] == False]
            df_true = df[df['funded'] == True]
            df_true_sub = df_true.sample(n = num_false)
            df_balance = pd.concat([df_true_sub, df_false])
            return(df_balance)
        self.df_train = downsample_majority(self.df_train)

    def X_Y_split(self, y_col = 'funded'):
        def df_to_X(df):
            return(df.drop(y_col, axis = 1))
        def df_to_Y(df):
            return(df[y_col])
        self.X_train = df_to_X(self.df_train)
        self.Y_train = df_to_Y(self.df_train)
        self.X_test = df_to_X(self.df_test)
        self.Y_test = df_to_Y(self.df_test)
