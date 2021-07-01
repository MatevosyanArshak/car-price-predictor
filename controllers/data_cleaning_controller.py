import pandas as pd
import numpy as np

class CleanData:

    def __init__(self, df):
        self.df = df

    def cleaning_sequence(self, df_column):
        self.get_important_col()
        self.drop_na()
        self.clean_IQR(df_column)

    def clean_IQR(self, df_column):
        for i in df_column:
            Q1 = self.df[i].quantile(.25)
            Q3 = self.df[i].quantile(.75)
            IQR = Q3 - Q1
            r = 1.5 * IQR
            outliers_indexes = self.df[(self.df[i] < (Q1 - r)) | (self.df[i] > (Q3 + r))].index
            self.df = self.df.drop(outliers_indexes, axis=0)

    def get_important_col(self):
        self.df = self.df[['price', 'year', 'odometer']]

    def drop_na(self):
        self.df.replace(0, np.inf, inplace=True)
        pd.set_option('use_inf_as_na', True)
        self.df = self.df.dropna()

    def clean_with_std(self):
        upper_limit = self.df.height.mean() + 3*self.df.height.std()
        lower_limit = self.df.height.mean() - 3*self.df.height.std()
        self.df = self.df[(self.df.height > upper_limit) | (self.df.height < lower_limit)]

    def get_dataframe(self):
        return self.df
