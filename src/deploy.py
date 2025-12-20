import os
import joblib
from typing import Any


class ModelDeployer:
    """
    Class for saving and loading models
    """

    def __init__(self, save_path: str):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def save_model(self, model: Any, filename: str):
        """
        Saves the model to the file in .pkl format
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        full_path = os.path.join(self.save_path, filename)

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

        full_path = os.path.join(self.save_path, filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")

        try:
            model = joblib.load(full_path)
            print(f"Model loaded from: {full_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {filename}: {e}")