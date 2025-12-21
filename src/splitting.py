import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple


class StratifiedSplitter:
    """
    Class for global reshuffling of the data for segments
    """

    def __init__(self, target_col: str, segment_cols: List[str], min_test_samples: int = 50):
        self.target_col = target_col
        self.segment_cols = segment_cols
        self.min_test_samples = min_test_samples

    def split_globally(self, df_train: pd.DataFrame, df_test: pd.DataFrame, test_size: float = 0.2,
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        1. Concats train and test.
        2. Creates stratify_key = SegmentCols + Target.
        3. Makes stratified distriburion.
        """
        print("--- Global Stratified Split Initiated ---")

        df_train['_origin'] = 'train'
        df_test['_origin'] = 'test'

        full_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
        print(f"   Merged dataset shape: {full_df.shape}")

        stratify_key = full_df[self.target_col].astype(str)

        for col in self.segment_cols:
            if col in full_df.columns:
                stratify_key = stratify_key + "_" + full_df[col].astype(str)
            else:
                print(f"   [WARNING] Segment column '{col}' missing for stratification key!")

        key_counts = stratify_key.value_counts()
        rare_keys = key_counts[key_counts < 2].index

        if len(rare_keys) > 0:
            print(
                f"   [WARNING] Found {len(rare_keys)} rare groups (single sample). They won't be stratified perfectly.")
            stratify_key = stratify_key.mask(stratify_key.isin(rare_keys), 'Other')

        new_train, new_test = train_test_split(
            full_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_key
        )

        cols_to_drop = ['_origin']
        new_train = new_train.drop(columns=cols_to_drop, errors='ignore')
        new_test = new_test.drop(columns=cols_to_drop, errors='ignore')

        print(f"   New Train shape: {new_train.shape}")
        print(f"   New Test shape:  {new_test.shape}")

        return new_train, new_test

    def validate_test_segments(self, df_test: pd.DataFrame, segment_configs: List[Dict]):
        """
        Checks if there are enough data in test
        """
        print("\n--- Validating Test Segments Size ---")
        issues_found = False

        for config in segment_configs:
            seg_name = config['name']
            filter_func = config['filter']

            try:
                subset = df_test[filter_func(df_test)]
                count = len(subset)

                if count < self.min_test_samples:
                    print(f"   [CRITICAL WARNING] Segment '{seg_name}' has only {count} samples in TEST set!")
                    print(f"                      Metrics for this segment will be statistically unreliable.")
                    issues_found = True
                else:
                    pass

            except Exception as e:
                print(f"   [Error] Could not validate segment '{seg_name}': {e}")

        if not issues_found:
            print("   All segments have sufficient testing data.")
        else:
            print("   Please consider merging small segments or increasing dataset size.")