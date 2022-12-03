import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class eda_processor:
    def __init__(self, df):
        self.df = df
        pass
    
    def drop_null_column(self, colName):
        return self.df.drop(columns=[colName])
    
    def labelEncode(self, colName):
        le = LabelEncoder()
        self.df[colName] = le.fit_transform(self.df[colName])
        return self.df
    
    def fluc_rate_cut(self, fluc_col='등락률'):
        self.df = self.df[abs(self.df[fluc_col]) <= 30]
        return self.df
    
    def get_data_per_stock(self, name_col='종목명', date_col='날짜', start_price_col='시가', end_price_col='종가'):
        outliers = []
        for item in self.df[name_col].drop_duplicates():
            a = self.df[self.df[name_col] == item][start_price_col].reset_index()
            b = self.df[self.df[name_col] == item][end_price_col].reset_index()
            if(a.loc[0][start_price_col] == 0) :
                outliers.append(item)
                
        data_per_stock = []
        for item in self.df[name_col].drop_duplicates():
            if(item in outliers):
                continue
        else:
            seriesData = self.df[self.df[name_col] == item]
            seriesData = seriesData.sort_values(by=[date_col])
            data_per_stock.append(seriesData)
        
        return data_per_stock
    
    def minmaxScale(self, data_per_stock, start_price_col='시가', end_price_col='종가', high_price_col='고가', low_price_col='저가', volume_col='거래량', trading_value_col='거래대금', capital_col='시가총액'):
        for i in range(len(data_per_stock)):
            scaler_start = MinMaxScaler()
            scaler_end = MinMaxScaler()
            scaler_low = MinMaxScaler()
            scaler_high = MinMaxScaler()
            scaler_volume = MinMaxScaler()
            scaler_tradingvalue = MinMaxScaler()
            scaler_capitalization = MinMaxScaler()
            data_per_stock[i][start_price_col] = scaler_start.fit_transform(data_per_stock[i]['시가'].reset_index()).T[1].T
            data_per_stock[i][end_price_col] = scaler_end.fit_transform(data_per_stock[i]['종가'].reset_index()).T[1].T
            data_per_stock[i][high_price_col] = scaler_high.fit_transform(data_per_stock[i]['고가'].reset_index()).T[1].T
            data_per_stock[i][low_price_col] = scaler_low.fit_transform(data_per_stock[i]['저가'].reset_index()).T[1].T
            data_per_stock[i][volume_col] = scaler_volume.fit_transform(data_per_stock[i]['거래량'].reset_index()).T[1].T
            data_per_stock[i][trading_value_col] = scaler_tradingvalue.fit_transform(data_per_stock[i]['거래대금'].reset_index()).T[1].T
            data_per_stock[i][capital_col] = scaler_capitalization.fit_transform(data_per_stock[i]['시가총액'].reset_index()).T[1].T
            
        return data_per_stock
    
    def get_xy(self, data_per_stock):
        X_for_dps = []
        y_for_dps = []

        for i in range(len(data_per_stock)):
            X_for_dps.append(data_per_stock[i][0:len(data_per_stock[i])-2])
            y_for_dps.append(data_per_stock[i][1:len(data_per_stock[i])-1][["종가", "시가", "고가", "저가"]])
            
        return X_for_dps, y_for_dps

    def train_test_split(self, X_for_dps, y_for_dps):
        X_train_dps, X_test_dps, y_train_dps, y_test_dps = train_test_split(X_for_dps, y_for_dps, test_size=0.33, random_state=42)
        
        return X_train_dps, X_test_dps, y_train_dps, y_test_dps
