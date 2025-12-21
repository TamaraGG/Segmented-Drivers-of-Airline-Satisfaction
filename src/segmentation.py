import pandas as pd
from typing import List, Dict, Callable, Tuple, Generator

from sklearn.model_selection import train_test_split


class SegmentManager:
    """
    Класс-оркестратор для управления сегментацией данных.
    Отвечает за:
    1. Фильтрацию данных по правилам (Config).
    2. Проверку размера выборки (Sanity check).
    3. Удаление константных колонок (Cleaning).
    4. Разделение на X и y.
    """

    def __init__(self, segment_configs: List[Dict], target_col: str, min_samples: int = 100):
        """
        :param segment_configs: list of config dictionaries.
                                the structure: {
                                    'name': str,
                                    'filter': function(df) -> bool/mask,
                                    'drop_cols': list[str]
                                }
        :param target_col: y.
        :param min_samples: minimal number of rows for segment training.
        """
        self.configs = segment_configs
        self.target_col = target_col
        self.min_samples = min_samples

    def _prepare_subset(self, df: pd.DataFrame, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Drops columns and separates target.
        """
        # columns to drop
        cols_to_remove = list(set(drop_cols + [self.target_col]))

        X = df.drop(columns=cols_to_remove, errors='ignore')

        y = df[self.target_col] if self.target_col in df.columns else None

        return X, y

    def _log_target_stats(self, y: pd.Series, seg_name: str):
        """
        Logs distribution of target variable
        """
        if y is None: return

        counts = y.value_counts()
        fracs = y.value_counts(normalize=True)

        print(f"   Target Distribution ({seg_name}):")
        for cls_val in counts.index:
            print(f"     Class {cls_val}: {counts[cls_val]:>5} samples ({fracs[cls_val]:6.1%})")

    def _log_target_stats(self, y: pd.Series, seg_name: str):
        """
        Logs statistics of target variable distribution.
        """
        if y is None:
            return

        counts = y.value_counts()
        fracs = y.value_counts(normalize=True)

        print(f"   Target Distribution ({seg_name}):")
        for cls_val in counts.index:
            print(f"     Class {cls_val}: {counts[cls_val]:>5} samples ({fracs[cls_val]:6.1%})")

    def _log_split_stats(self, train_df: pd.DataFrame, test_df: pd.DataFrame, seg_name: str):
        """
        Logs sizes of Train and Test data.
        """
        n_train = len(train_df)
        n_test = len(test_df) if test_df is not None else 0
        total = n_train + n_test

        print(f"   Data Split Sizes ({seg_name}):")
        if total == 0:
            print("     Total: 0 samples (Empty)")
            return

        print(f"     Total: {total}")
        print(f"     Train: {n_train:>5} ({n_train / total:6.1%})")

        if n_test > 0:
            print(f"     Test:  {n_test:>5} ({n_test / total:6.1%})")
        else:
            print(f"     Test:      0 (No test data)")

    def _is_segment_viable(self, y: pd.Series, seg_name: str) -> bool:
        """
        Desids weather it's possible to train the model with such distribution of target variable
        """
        if y is None: return True

        counts = y.value_counts()

        if len(counts) < 2:
            print(f"   [SKIP] Training impossible: Segment contains only Class {counts.index[0]}.")
            return False

        min_class_count = counts.min()
        if min_class_count < 15:
            print(f"   [WARNING] Risk of overfitting: Minority class has only {min_class_count} samples.")

        return True

    def _ensure_balanced_split(self, train_df: pd.DataFrame, test_df: pd.DataFrame, seg_name: str) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Checks weather there are enough data in test and train data.
        Returns train and test data with correct distribution.
        """

        if test_df is None:
            return train_df, None

        n_train = len(train_df)
        n_test = len(test_df)
        total = n_train + n_test

        if total < self.min_samples:
            return train_df, test_df

        min_test_size = 20

        if n_test < min_test_size or n_train < self.min_samples:
            print(f"   [RE-SPLIT] Bad original split (Train:{n_train}, Test:{n_test}). Reshuffling {total} samples...")

            full_df = pd.concat([train_df, test_df])

            try:
                new_train, new_test = train_test_split(
                    full_df,
                    test_size=0.2,
                    random_state=42,
                    stratify=full_df[self.target_col]
                )
            except ValueError:
                print("   [WARNING] Stratified split failed (rare class). Using random split.")
                new_train, new_test = train_test_split(
                    full_df,
                    test_size=0.2,
                    random_state=42
                )

            return new_train, new_test

        return train_df, test_df

    def _check_distribution(self, y: pd.Series, seg_name: str) -> bool:
        """
        Checks distribution of target feature in a segment.
        """
        if y is None:
            return True

        counts = y.value_counts()
        fracs = y.value_counts(normalize=True)

        print(f"   Target Distribution (y_train):")
        for cls_val in counts.index:
            print(f"     Class {cls_val}: {counts[cls_val]} samples ({fracs[cls_val]:.1%})")

        if len(counts) < 2:
            print(f"   [SKIP] Segment '{seg_name}' has only ONE target class. Training impossible.")
            return False

        min_class_count = counts.min()
        if min_class_count < 10:
            print(f"   [WARNING] Extremely rare minority class ({min_class_count} samples). Results might be unstable.")

        return True

    def iterate_segments(self, df_train: pd.DataFrame, df_test: pd.DataFrame = None) -> Generator:
        """
        Generator that generates data for every segment.
        Yields:
            tuple: (segment_name, X_train, y_train, X_test, y_test)
        """
        print(f"Starting segmentation process. Total configurations: {len(self.configs)}")

        for config in self.configs:
            seg_name = config['name']
            filter_func = config['filter']
            drop_cols = config.get('drop_cols', [])

            # 1. filter
            try:
                train_mask = filter_func(df_train)
                train_subset = df_train[train_mask].copy()

                test_subset = None
                if df_test is not None:
                    test_mask = filter_func(df_test)
                    test_subset = df_test[test_mask].copy()



            except Exception as e:
                print(f"Error filtering segment '{seg_name}': {e}")
                continue

            # 2. validation

            if len(train_subset) < self.min_samples:
                print(f"SKIPPING segment '{seg_name}': Not enough samples ({len(train_subset)} < {self.min_samples})")
                continue

            print(f"\nProcessing segment: {seg_name}")
            print(f"   Train shape: {train_subset.shape}")
            if test_subset is not None:
                print(f"   Test shape:  {test_subset.shape}")

            # 3. feature selection
            X_train, y_train = self._prepare_subset(train_subset, drop_cols)

            #self._log_split_stats(train_subset, test_subset, seg_name)
            #self._log_target_stats(y_train, seg_name)

            if not self._is_segment_viable(y_train, seg_name):
                continue

            X_test, y_test = None, None
            if test_subset is not None:
                X_test, y_test = self._prepare_subset(test_subset, drop_cols)

            # 4. result
            yield {
                'name': seg_name,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }

        print("\nSegmentation finished.")