import pandas as pd
from typing import List, Dict, Callable, Tuple, Generator


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