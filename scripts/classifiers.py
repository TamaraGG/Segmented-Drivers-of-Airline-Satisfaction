from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import joblib
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


class KNNModel:
    def __init__(self):
        # Hyperparameters definition
        self.params = {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],  # Distance-based weighting
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "metric": ["euclidean", "manhattan"]
        }

    def train(self, x_train, y_train, cv=5, scoring="accuracy"):
        # Define the model
        model = KNeighborsClassifier()

        # Hyperparameter optimization
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.params,
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )
        grid_search.fit(x_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        print("Best KNN Parameters:", grid_search.best_params_)
        
        # Return best model and grid search results for further analysis
        return best_model, grid_search.cv_results_

    def get_best_params(self):
        """Returns the best hyperparameters."""
        if self.best_model is None:
            raise ValueError("The model is not trained yet. Call train() first.")
        return self.best_model.get_params()

    def get_cv_results(self):
        """Returns a DataFrame with the cross-validation results."""
        if self.cv_results_ is None:
            raise ValueError("The model is not trained yet. Call train() first.")
        return self.cv_results_


# Random Forest Class
class RandomForestModel:
    def __init__(self):
        # Hyperparameters definition
        self.params = {
            "n_estimators": [10, 50, 100],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }

    def train(self, x_train, y_train):
        # Define the model
        model = RandomForestClassifier(class_weight="balanced", random_state=42)
        
        # Hyperparameter optimization
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.params,
                                   scoring="accuracy",
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        print("Best Random Forest Parameters:", grid_search.best_params_)
        
        return best_model, grid_search.cv_results_

    def get_best_params(self):
        """Returns the best hyperparameters."""
        if self.best_model is None:
            raise ValueError("The model is not trained yet. Call train() first.")
        return self.best_model.get_params()

    def get_cv_results(self):
        """Returns a DataFrame with the cross-validation results."""
        if self.cv_results_ is None:
            raise ValueError("The model is not trained yet. Call train() first.")
        return self.cv_results_


# XGBoost Class
class XGBoostModel:
    def __init__(self):
        # Hyperparameters definition
        self.params = {
            "n_estimators": [20, 50, 100],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [8, 16, 32],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.8, 0.9]
        }

    def train(self, x_train, y_train):
        # Define the model
        model = xgb.XGBClassifier()

        # Hyperparameter optimization
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.params,
                                   scoring="accuracy",
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        print("Best XGBoost Parameters:", grid_search.best_params_)

        return best_model


# XGBoost Class for One Segment
class SegmentedXGBoost:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_estimator_ = None
        self.cv_results_ = None

        self.params = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "scale_pos_weight": [1]
        }

    def train(self, x_train, y_train):
        # 1. Base model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=1
        )

        # 2. Set up GridSearch
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.params,
            scoring="f1",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=-1,
            verbose=1
        )

        # 3. Start training
        print(f"Training XGBoost on {x_train.shape[0]} samples...")
        grid_search.fit(x_train, y_train)

        # 4. Save results inside the class
        self.best_estimator_ = grid_search.best_estimator_
        self.cv_results_ = grid_search.cv_results_

        print("Best XGBoost Parameters:", grid_search.best_params_)
        print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

        return self.best_estimator_

    def get_best_params(self):
        """Returns the best hyperparameters."""
        if self.best_estimator_ is None:
            raise ValueError("Model not trained.")
        return self.best_estimator_.get_params()

    def get_cv_results(self):
        """Returns the CV results dictionary."""
        if self.cv_results_ is None:
            raise ValueError("Model not trained.")
        return self.cv_results_