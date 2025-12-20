import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import List, Optional, Union


class ShapAnalyzer:
    """
    A class for interpreting models using SHAP.
    Allows you to plot feature importance, dependence, and interaction graphs.
    """

    def __init__(self, model, X_train: pd.DataFrame):
        """
        Initializes Explainer.

        :param model: trained model (XGBoost, CatBoost, Sklearn Tree).
        :param X_train: data used to train the model (нужны для инициализации Explainer).
        """
        self.model = model

        self.explainer = shap.TreeExplainer(self.model)

        self._shap_values_cache = None
        self._X_cache = None

    def _get_shap_values(self, X: pd.DataFrame):
        """
        Calculates or takes from the cache SHAP values.
        """

        if self._shap_values_cache is None or not X.equals(self._X_cache):
            self._X_cache = X.copy()
            self._shap_values_cache = self.explainer(X)

        return self._shap_values_cache

    def get_top_features(self, X: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Returns DataFrame with top N values (Mean Absolute SHAP).
        """
        shap_values = self._get_shap_values(X)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

        df_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': mean_abs_shap
        }).sort_values(by='Importance', ascending=False)

        return df_imp.head(top_n)

    def plot_summary(self, X: pd.DataFrame, max_display: int = 15, plot_type: str = "dot"):
        """
        Builds Summary Plot (Beeswarm).
        Shows impact importance and direction.
        """
        shap_values = self._get_shap_values(X)

        plt.figure(figsize=(10, 6))
        plt.title("SHAP Summary Plot", fontsize=14)

        shap.summary_plot(
            shap_values,
            X,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.show()

    def plot_dependence(self, X: pd.DataFrame, feature: str, interaction_feature: str = None):
        """
        Builds plot for Saturation Analysis.
        Shows how specific feature value impacts prediction.

        :param feature: feature for analysis (e.g. 'Inflight wifi service').
        :param interaction_feature: optional param.
        """
        if feature not in X.columns:
            print(f"Skipping plot: Feature '{feature}' not found in data.")
            return

        shap_values = self._get_shap_values(X)

        plt.figure(figsize=(8, 5))

        shap.plots.scatter(
            shap_values[:, feature],
            color=shap_values[:, interaction_feature] if interaction_feature else shap_values,
            x_jitter=0.5,
            title=f"Dependence Plot: {feature}",
            show=False
        )

        plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_interaction(self, X: pd.DataFrame, feature_x: str, feature_color: str):
        """
        Builds plot of two features interaction.
        """
        if feature_x not in X.columns or feature_color not in X.columns:
            print(f"Skipping interaction: Features '{feature_x}' or '{feature_color}' missing.")
            return

        shap_values = self._get_shap_values(X)

        plt.figure(figsize=(9, 6))

        shap.plots.scatter(
            shap_values[:, feature_x],
            color=shap_values[:, feature_color],
            title=f"Interaction: Does [{feature_color}] affect [{feature_x}] impact?",
            show=False
        )

        plt.tight_layout()
        plt.show()