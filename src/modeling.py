import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from typing import Dict, Any, Tuple


class ModelTrainer:
    """
    A class for training and tuning XGBoost models.
    Automatically handles class imbalances and performs GridSearch.
    """

    def __init__(self, fixed_params: Dict[str, Any] = None, random_state: int = 42):
        self.random_state = random_state
        # default params
        self.fixed_params = fixed_params if fixed_params else {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'tree_method': 'hist'
        }

        self.fixed_params['random_state'] = self.random_state

        self.best_estimator_ = None
        self.best_params_ = None
        self.cv_results_ = None

    def _calculate_scale_weight(self, y: pd.Series) -> float:
        """
        Counts coefficients for class balancing
        Weight = Negative / Positive
        """
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    def train(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, list],
              scoring: str = "f1") -> xgb.XGBClassifier:
        """
        GridSearch with cross validation.

        :param X: feature matrix (numeric).
        :param y: target value.
        :param param_grid: dict with params.
        :param scoring: for optimization (f1, roc_auc, accuracy).
        :return: best model.
        """
        print(f"   >>> Training XGBoost on {len(X)} samples...")

        # 1. class weight
        calc_weight = self._calculate_scale_weight(y)

        # 2. params grid
        search_grid = param_grid.copy()

        if 'scale_pos_weight' not in search_grid:
            search_grid['scale_pos_weight'] = [1.0, calc_weight]
            search_grid['scale_pos_weight'] = sorted(list(set(search_grid['scale_pos_weight'])))

        # 3. base model
        model = xgb.XGBClassifier(**self.fixed_params)

        # 4. cross-validation setup
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # 5. Grid Search
        gs = GridSearchCV(
            estimator=model,
            param_grid=search_grid,
            scoring=scoring,
            cv=cv_strategy,
            n_jobs=-1,
            verbose=1  # 1 for debug
        )

        try:
            gs.fit(X, y)
        except Exception as e:
            print(f"   !!! Training Failed: {e}")
            raise e

        # 6. save results
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.cv_results_ = gs.cv_results_

        print(f"   >>> Best Params: {gs.best_params_}")
        print(f"   >>> Best CV {scoring.upper()}: {gs.best_score_:.4f}")

        return self.best_estimator_

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns DataFrame with feature importance (Gain) for trained model.
        """
        if self.best_estimator_ is None:
            raise ValueError("Model is not trained yet.")

        importance = self.best_estimator_.feature_importances_
        features = self.best_estimator_.feature_names_in_

        df_imp = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        return df_imp

    def get_best_params(self) -> Dict:
        """Returns the best hyperparameters."""
        if self.best_params_ is None:
            raise ValueError("Model is not trained yet. Call train() first.")
        return self.best_params_

    def get_all_params(self) -> Dict:
        """Returns all hyperparameters."""
        if self.best_estimator_ is None:
            raise ValueError("Model is not trained yet. Call train() first.")
        return self.best_estimator_.get_params()

    def get_cv_results(self) -> Dict:
        """Returns the CV results dictionary."""
        if self.cv_results_ is None:
            raise ValueError("Model is not trained yet. Call train() first.")
        return self.cv_results_

    def get_best_model(self) -> xgb.XGBClassifier:
        """Returns the model."""
        if self.best_estimator_ is None:
            raise ValueError("Model is not trained yet. Call train() first.")
        return self.best_estimator_