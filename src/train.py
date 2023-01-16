import os
import sys
import pandas as pd
import numpy as np
import argparse

sys.path.append('..')
from src import mylogger
from src import preprocessing
from src import model_xgboost

ROOT_PATH = os.getcwd().replace('src', '')

class Train:
    def __init__(
        self, 
        xgboost: bool
    ):
        self._xgboost = xgboost
        self.logger = mylogger.Logger(program_name='train').initialize_logger()

    def train_process(
        self
    ):
        # Load dataset.
        df = pd.read_csv(ROOT_PATH + 'data/data.csv')

        # Create index of training and validation.
        df_index = df.index
        df_length = int(len(df_index))
        train_val_index = df_index[:int(df_length * 0.8)]
        train_index = train_val_index[:int(len(train_val_index) * 0.8)]
        val_index = train_val_index[int(len(train_val_index) * 0.8):]
        test_index = df_index[int(df_length * 0.8):]

        if self._xgboost:
            self.logger.info('XGBoost training start.')

            # Preprocessing.
            self.prepro = preprocessing.TrainingPreprocessing()
            preprocessed_df = self.prepro.preprocessing(df)

            # Split explaining variable and target variable.
            target = preprocessed_df['close']
            preprocessed_df = preprocessed_df['open']

            # Training.
            cb = model_xgboost.XGBoost(
                preprocessed_df.iloc[train_index], 
                preprocessed_df.iloc[val_index], 
                preprocessed_df.iloc[test_index], 
                target.iloc[train_index], 
                target.iloc[val_index], 
                target.iloc[test_index]
            )

if __name__ == '__main__':
    # Instance of argparse.
    parser = argparse.ArgumentParser()

    # Add argument to parser.
    parser.add_argument('--xgboost', action='store_true')    # [python3 train.py --xgboost] means [python3 train.py --xgboost True] at terminal by using 'action="store_true"'.
    args = parser.parse_args()

    train = Train(
        xgboost=args.xgboost
    )
    train.train_process()
