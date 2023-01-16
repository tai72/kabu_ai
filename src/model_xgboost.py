import sys
import json
import optuna
import xgboost as xgb
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

sys.path.append('../')
from src import gcs_ex

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

        # Read settings file.
        try:
            config = json.load(open('../meta/settings.json', "r", encoding="utf-8"))
        except FileNotFoundError as e:
            print("[ERROR] Config file is not found.")
            raise e
        except ValueError as e:
            print("[ERROR] Json file is invalid.")
            raise e
        
        # Bucket
        self.bucket = gcs_ex.GCSBucket(config['project_name'], config['bucket_name'])
    
    def objective(
        self, 
        trial
    ) -> float:
        trains = xgb.DMatrix(self.preprocessed_df_train, label=self.target_train)
        valids = xgb.DMatrix(self.preprocessed_df_val, label=self.target_val)

        params = {
            "silent": 1, 
            "max_depth": trial.suggest_int("max_depth", 4, 10), 
            'min_child_weight':trial.suggest_int('min_child_weight',1,5), 
            "eta": trial.suggest_loguniform("eta", 0.01, 1.0), 
            'gamma':trial.suggest_uniform('gamma',0,1), 
            'subsample':trial.suggest_uniform('subsample',0,1), 
            'colsample_bytree':trial.suggest_uniform('colsample_bytree',0,1), 
            'reg_alpha':trial.suggest_loguniform('reg_alpha',1e-5,100), 
            'reg_lambda':trial.suggest_loguniform('reg_lambda',1e-5,100), 
            'learning_rate':trial.suggest_uniform('learning_rate',0,1), 
            "tree_method": "exact", 
            "objective": "reg:linear", 
            "eval_metric": "rmse", 
            "predictor": "cpu_predictor"
        }

        model = xgb.train(
            params, trains, 
            num_boost_round=1000, 
            early_stopping_rounds=50, 
            evals=[(valids, 'valid')]
        )
        preds = model.predict(xgb.DMatrix(self.preprocessed_df_test))
        rmse = metrics.mean_squared_error(self.target_test, preds)

        return rmse
    
    def train_process(
        self
    ) -> optuna.study:
        # Search optimized parameter.
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=100)

        # Create model by using best parameters.
        trains = xgb.DMatrix(self.preprocessed_df_train, self.target_train)
        tests = xgb.DMatrix(self.preprocessed_df_test, self.target_test)
        model = xgb.train(
            study.best_params, trains, 
            num_boost_round=1000, 
            early_stopping_rounds=50, 
            evals=[(tests, 'test')]
        )

        # Predict
        preds = model.predict(xgb.DMatrix(self.preprocessed_df_test))

        # Evaluation
        rmse = metrics.mean_squared_error(self.target_test, preds)
        r2 = metrics.r2_score(self.target_test, preds)

        # Visualize
        df_test = pd.DataFrame(self.target_test)
        df_test['close_pred'] = preds
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.plot(df_test['close'], color='#ea5972', label='true')
        ax.plot(df_test['close_pred'], color='#005872', label='pred')
        ax.legend()
        ax.text(0.05, 0.9, f'rmse: {rmse}', transform=ax.transAxes, fontsize=10)
        ax.text(0.05, 0.85, f'R2: {r2}', transform=ax.transAxes, fontsize=10)
        plt.savefig('../img/xgboost_result.png')

        # Output model.
