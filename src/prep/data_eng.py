# data_eng.py
import pandas as pd
import numpy as np
import re

class Validator:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def check_invalids(self, patterns=None):
        """
        Detect:
        - null values
        - invalid values (regex + numeric coercion)

        patterns: dict (optional)
        {
            "column_name": [regex1, regex2],
            "_global": [regex1, regex2]
        }
        """

        if patterns is None:
            patterns = {
                "_global": [
                    r"^\s*$",          # empty / whitespace
                    r"^\?$",           # ?
                    r"^(na|n/a|null|unknown)$"  # common unknowns
                ]
            }

        report = {
            "nulls": self.data.isnull().sum().to_dict(),
            "invalids": {}
        }

        for col in self.data.columns:
            series = self.data[col].astype(str)

            # --- combine global + column-specific patterns
            col_patterns = patterns.get("_global", []) + patterns.get(col, [])

            # --- regex unknown detection
            unknown_mask = pd.Series(False, index=series.index)
            for pat in col_patterns:
                unknown_mask |= series.str.lower().str.match(pat)

            # --- numeric coercion check (applies to ALL columns now)
            coerced = pd.to_numeric(series, errors='coerce')

            invalid_mask = (
                coerced.isna() &
                self.data[col].notna() &
                ~unknown_mask
            )

            # --- update report
            report["invalids"][col] = int(invalid_mask.sum())
            report["nulls"][col] += int(unknown_mask.sum())

        return report

    def column_types(self):
        """
        Infer column types
        """
        col_types = {}

        for col in self.data.columns:
            dtype = self.data[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                col_types[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_types[col] = "datetime"
            else:
                col_types[col] = "categorical"

        return col_types

    def outlier_detection(self, method="iqr"):
        """
        Detect outliers using:
        - IQR (default)
        - Z-score
        """
        outliers = {}

        numeric_cols = self.data.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            series = self.data[col].dropna()

            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                mask = (series < lower) | (series > upper)

            elif method == "zscore":
                mean = series.mean()
                std = series.std()

                if std == 0:
                    mask = pd.Series([False] * len(series), index=series.index)
                else:
                    z = (series - mean) / std
                    mask = np.abs(z) > 3

            else:
                raise ValueError("Unsupported method")

            outliers[col] = series[mask].index.tolist()

        return outliers


class Filler:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def _get_fill_value(self, series, method: str):
        if pd.api.types.is_numeric_dtype(series):
            if method == "mean":
                return series.mean()
            elif method == "median":
                return series.median()
            elif method == "mode":
                return series.mode().iloc[0]
            elif method == "std":
                return series.std()
            else:
                raise ValueError("Unsupported method")
        else:
            return series.mode().iloc[0]

    def fill(self, method: str = "mean"):
        df = self.data.copy()

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            value = self._get_fill_value(df[col], method)
            df[col] = df[col].fillna(value)

        return df

    def drop_identifier_columns(self, keywords=None):
        df = self.data.copy()

        if keywords is None:
            keywords = ["id", "key", "serial"]

        cols_to_drop = []

        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in keywords):
                cols_to_drop.append(col)

        df = df.drop(columns=cols_to_drop)

        return df
    
    def handle_invalids(
        self,
        method: str = "mean",
        drop_invalid: bool = False,
        rules: dict = None
    ):
        """
        Handle invalid values:
        - If drop_invalid=True → drop rows containing invalids
        - Else → replace invalids using fill strategy

        rules (optional):
        {
            "column_name": lambda x: True if valid else False
        }
        """
        df = self.data.copy()

        invalid_mask_total = pd.Series(False, index=df.index)

        for col in df.columns:
            series = df[col]

            # --- Rule-based validation (if provided)
            if rules and col in rules:
                valid_mask = series.apply(rules[col])
                invalid_mask = ~valid_mask

            # --- Default numeric coercion check
            elif pd.api.types.is_numeric_dtype(series):
                coerced = pd.to_numeric(series, errors='coerce')
                invalid_mask = coerced.isna() & series.notna()

            else:
                # assume categorical is valid unless rules provided
                invalid_mask = pd.Series(False, index=df.index)

            # accumulate invalid rows
            invalid_mask_total |= invalid_mask

            if not drop_invalid:
                fill_value = self._get_fill_value(series[~invalid_mask], method)
                df.loc[invalid_mask, col] = fill_value

        if drop_invalid:
            df = df.loc[~invalid_mask_total]

        return df