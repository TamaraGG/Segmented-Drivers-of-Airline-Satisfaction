import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Tuple


class ModelEvaluator:
    """
    A class for assessing the quality of classification models.
    Generates metrics, reports, and graphs.
    """

    def __init__(self):
        pass

    def calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series = None) -> Dict[str, float]:
        """
        Counts metrics and returns dict.
        """
        # base metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # specificity (True Negative Rate)
        # TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Specificity": round(specificity, 4),
            "F1-Score": round(f1, 4)
        }

        # ROC-AUC
        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
                metrics["ROC_AUC"] = round(roc_auc, 4)
            except ValueError:
                metrics["ROC_AUC"] = np.nan

        return metrics

    def print_report(self, y_true: pd.Series, y_pred: pd.Series, segment_name: str = "Segment"):
        """
        Prints standard text sklearn report.
        """
        print(f"\n{'=' * 20} Classification Report: {segment_name} {'=' * 20}")
        print(classification_report(y_true, y_pred, digits=4))

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series, title: str = "Confusion Matrix",
                              classes: List = [0, 1]):
        """
        Plots confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        plt.figure(figsize=(6, 5))
        sbn.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['True 0', 'True 1'])

        plt.title(title, fontsize=14)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_segments(metrics_storage: Dict[str, Dict]) -> pd.DataFrame:
        """
        Accepts a dictionary with metrics for all segments and collects them
        into a single Pandas table.

        :param metrics_storage: {'Segment1': {'Accuracy': 0.9, ...}, 'Segment2': ...}
        """
        df_comparison = pd.DataFrame.from_dict(metrics_storage, orient='index')

        if 'F1-Score' in df_comparison.columns:
            df_comparison = df_comparison.sort_values(by='F1-Score', ascending=False)

        return df_comparison