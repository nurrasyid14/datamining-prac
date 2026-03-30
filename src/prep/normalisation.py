import pandas as pd
import numpy as np

class Normalisation:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def zscore(self, use_scikit: bool = False):
        df = self.data.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns

        if use_scikit:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        else:
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()

                if std == 0:
                    df[col] = 0  # avoid division by zero
                else:
                    df[col] = (df[col] - mean) / std

        return df

    def minmax(self, use_scikit: bool = True):
        df = self.data.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns

        if use_scikit:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()

            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        else:
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()

                if max_val - min_val == 0:
                    df[col] = 0
                else:
                    df[col] = (df[col] - min_val) / (max_val - min_val)

        return df