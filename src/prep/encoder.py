import pandas as pd
import numpy as np


import pandas as pd
import numpy as np


class Encoder:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def to_numeric(self, column, errors: str = "coerce"):
        df = self.data.copy()
        df[column] = pd.to_numeric(df[column], errors=errors)
        return df

    def to_categorical(self, column):
        df = self.data.copy()
        df[column] = df[column].astype("category")
        return df

    def ohe(self, column, drop_first: bool = False, dummy_na: bool = False):
        df = self.data.copy()

        dummies = pd.get_dummies(
            df[column],
            prefix=column,
            drop_first=drop_first,
            dummy_na=dummy_na
        )

        df = df.drop(columns=[column])
        df = pd.concat([df, dummies], axis=1)

        return df

    def binary_encode(
        self,
        column,
        true_values=None,
        false_values=None,
        casefold: bool = True
    ):
        """
        Convert binary-like categorical values to 0/1.

        Default:
        yes → 1
        no  → 0
        """
        df = self.data.copy()

        if true_values is None:
            true_values = ["yes", "y", "true", "1","ya",]

        if false_values is None:
            false_values = ["no", "n", "false", "0","tak","tidak",]

        series = df[column]

        if casefold:
            series = series.astype(str).str.lower()

        def mapper(x):
            if x in true_values:
                return 1
            elif x in false_values:
                return 0
            else:
                return np.nan  # unknown values

        df[column] = series.map(mapper)

        return df