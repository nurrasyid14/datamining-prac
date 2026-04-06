import pandas as pd
import numpy as np

class Normalisation:
    def __init__(self, method: str = "minmax", use_scikit: bool = False):
        self.method = method.lower()
        self.use_scikit = use_scikit

        self.numeric_cols = None

        # parameters
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.std_ = None

        # scikit
        self.scaler = None

    def fit(self, data: pd.DataFrame):
        self.numeric_cols = data.select_dtypes(include=np.number).columns

        if self.use_scikit:
            if self.method == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
            elif self.method == "zscore":
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            else:
                raise ValueError("Sigmoid not supported in scikit")

            self.scaler.fit(data[self.numeric_cols])

        else:
            if self.method == "minmax":
                self.min_ = data[self.numeric_cols].min()
                self.max_ = data[self.numeric_cols].max()

            elif self.method == "zscore":
                self.mean_ = data[self.numeric_cols].mean()
                self.std_ = data[self.numeric_cols].std()

            elif self.method == "sigmoid":
                self.mean_ = data[self.numeric_cols].mean()
                self.std_ = data[self.numeric_cols].std()

            else:
                raise ValueError("Invalid method")

    def transform(self, data: pd.DataFrame):
        df = data.copy()

        if self.use_scikit:
            df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
            return df

        for col in self.numeric_cols:
            if self.method == "minmax":
                min_val = self.min_[col]
                max_val = self.max_[col]

                if max_val - min_val == 0:
                    df[col] = 0
                else:
                    df[col] = (df[col] - min_val) / (max_val - min_val)

            elif self.method == "zscore":
                mean = self.mean_[col]
                std = self.std_[col]

                if std == 0:
                    df[col] = 0
                else:
                    df[col] = (df[col] - mean) / std

            elif self.method == "sigmoid":
                mean = self.mean_[col]
                std = self.std_[col]

                if std == 0:
                    z = 0
                else:
                    z = (df[col] - mean) / std

                df[col] = 1 / (1 + np.exp(-z))

        return df

    def fit_transform(self, data: pd.DataFrame):
        self.fit(data)
        return self.transform(data)