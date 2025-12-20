import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os


class GlobalEncoder:
    """
    A class for global encoding of categorical features.
    Supports:
    1. Manual encoding (mapping) with order preservation.
    2. Automatic encoding (OrdinalEncoder) for other categories.
    3. Creating new columns with a suffix (to preserve the original columns for filtering).
    """

    def __init__(self, manual_mappings: dict = None, auto_cols: list = None):
        """
        :param manual_mappings: dictionary (ex. {'ColName': {'Map': {'Eco':0...}, 'Suffix': '_Encoded'}} )
                                Suffix is optional. If there is no suffix rewrite values in column.
        :param auto_cols: list of columns for auto OrdinalEncoding (Gender, Type...).
        """
        self.manual_mappings = manual_mappings if manual_mappings else {}
        self.auto_cols = auto_cols if auto_cols else []

        self.auto_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=int
        )
        self._is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Trains auto encoder for given dataframe."""
        if self.auto_cols:
            # Проверяем наличие колонок
            valid_cols = [c for c in self.auto_cols if c in df.columns]
            if len(valid_cols) != len(self.auto_cols):
                missing = set(self.auto_cols) - set(valid_cols)
                print(f"Warning: Columns {missing} not found in DF during fit.")

            if valid_cols:
                self.auto_encoder.fit(df[valid_cols])
                self._is_fitted = True
                print(f"GlobalEncoder fitted on {len(valid_cols)} auto-columns.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding for dataframe."""
        df = df.copy()

        # 1. manual mapping
        for col, config in self.manual_mappings.items():
            if col in df.columns:
                mapping_dict = config.get('Map')
                suffix = config.get('Suffix', '')

                target_col = f"{col}{suffix}"

                df[target_col] = df[col].map(mapping_dict)

                df[target_col] = df[target_col].fillna(-1).astype(int)

                action = "Created" if suffix else "Overwrote"
                print(f"{action} column '{target_col}' using manual mapping.")

        # 2. auto encoder
        if self.auto_cols and self._is_fitted:
            valid_cols = [c for c in self.auto_cols if c in df.columns]
            if valid_cols:
                encoded_data = self.auto_encoder.transform(df[valid_cols])
                df[valid_cols] = encoded_data

        elif self.auto_cols and not self._is_fitted:
            raise ValueError("GlobalEncoder is not fitted! Call .fit() first.")

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper: train and fit."""
        self.fit(df)
        return self.transform(df)

    # ==========================================
    # Save / Load (for MLOps)
    # ==========================================
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Encoder saved to {path}")

    @staticmethod
    def load(path: str):
        return joblib.load(path)