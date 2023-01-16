import optuna
import xgboost as xgb
import pandas as pd
import numpy as np

class XGBoost:
    def __init__(
        self, 
        df_train: pd.DataFrame, 
        df_val: pd.DataFrame, 
        df_test: pd.DataFrame, 
        target_train: pd.Series, 
        target_val: pd.Series, 
        target_test: pd.Series
    ):
        self.preprocessed_df_train = df_train
        self.preprocessed_df_val = df_val
        self.preprocessed_df_test = df_test
        self.target_train = target_train
        self.target_val = target_val
        self.target_test = target_test
