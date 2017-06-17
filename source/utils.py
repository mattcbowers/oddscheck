# utility functions for the python scripts in source/
import pandas as pd

class ProjectsData(object, fname):

    def __init__(self):
        self.fname = fname

    def get_data(self, frac_na_tolerable = 0.05):
        # all col names
        col_names = ['_projectid', '_teacher_acctid', '_schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject', 'primary_focus_area' ,'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'vendor_shipping_charges', 'sales_tax', 'payment_processing_charges', 'fulfillment_labor_materials', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'funding_status', 'date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration']
        # col names to keep initially
        col_initial_keep = ['funding_status', 'school_state', 'school_metro', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_area', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'total_price_excluding_optional_support', 'students_reached']
        self = pd.read_csv(self.fname, escapechar='\\', names = col_names) 
        # convert number students to float
        self['date_posted'] = pd.to_datetime(self['date_posted'])
        self = self[self['date_posted'] < pd.to_datetime('2016-10-01')]
        print('removed the recent dates with incomplete projects')
        print(self.shape)
        self.students_reached = self.students_reached.astype(float)
        print("Initial size of data")
        print(self.shape)
        self = self[col_initial_keep]
        print("Size of data after removing unnecessary columns apriori")
        print(self.shape)
        frac_na = self.isnull().mean()
        # keep only columns that have less than 5% missing
        cols_to_keep = list(frac_na.loc[frac_na < frac_na_tolerable].index)
        self = self[cols_to_keep]
        print("The following columns have > 5% missing and are dropped")
        print(frac_na.loc[frac_na >= frac_na_tolerable].index)
        print(self.shape)
        # remove outliers
        price_cap = 10000
        students_reached_cap = 10000
        self = self[self['students_reached'] < students_reached_cap]
        self = self[self['total_price_excluding_optional_support'] < price_cap]
        print("Removing outliers")
        print(self.shape)
        # remove rows with NA
        self.dropna(inplace = True)
        print("Remove any rows with missing values")
        print(self.shape)
        # create binary funded status variable and remove the string version
        self['funded'] = self['funding_status'] == 'completed'
        self.drop('funding_status', axis = 1, inplace = True)
        return(self)
