import os
import joblib
from typing import Any, Dict
import json
import matplotlib.pyplot as plt

class ModelDeployer:
    """
    Class for saving and loading models
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.models_path = os.path.join(base_path, "models")
        self.reports_path = os.path.join(base_path, "reports")
        self.plots_path = os.path.join(base_path, "plots")
        self.params_path = os.path.join(base_path, "params")

        for p in [self.models_path, self.reports_path, self.plots_path, self.params_path]:
            os.makedirs(p, exist_ok=True)

    def save_model(self, model: Any, filename: str):
        """
        Saves the model to the file in .pkl format
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        full_path = os.path.join(self.models_path, f"{filename}.pkl")

        try:
            joblib.dump(model, full_path)
            print(f"   [Saved] Model saved to: {full_path}")
        except Exception as e:
            print(f"   [Error] Failed to save model {filename}: {e}")

    def load_model(self, filename: str) -> Any:
        """
        Loads .pkl file with the model
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        full_path = os.path.join(self.models_path, f"{filename}.pkl")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")

        try:
            model = joblib.load(full_path)
            print(f"Model loaded from: {full_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {filename}: {e}")

    def save_metrics(self, metrics: Dict, filename: str):
        """Saves metrics in JSON"""
        path = os.path.join(self.reports_path, f"{filename}_metrics.json")
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"   [Saved] Metrics: {path}")

    def save_plot(self, filename: str):
        """
        Saves current active plot (plt.gcf) in .png format.
        It should be called before plt.show()
        """
        path = os.path.join(self.plots_path, f"{filename}.png")
        plt.savefig(path, bbox_inches='tight', dpi=150)
        print(f"   [Saved] Plot: {path}")

    def save_params(self, params: Dict, filename: str):
        """

        :param params:
        :param filename:
        :return:
        """

        path = os.path.join(self.reports_path, f"{filename}_params.json")
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"   [Saved] Params: {path}")