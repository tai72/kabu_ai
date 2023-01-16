import pandas as pd
import numpy as np
import os

class Preprocessing:
    def __init__(
        self
    ):
        pass

    def liner_interpolation(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        df = df.interpolate()
        return df

class TrainingPreprocessing(Preprocessing):
    def preprocessing(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        df = self.liner_interpolation(df)

        return df
