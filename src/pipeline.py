import pandas as pd
import matplotlib.pyplot as plt
from src.interpretation import ShapAnalyzer
import src.config as cfg


class AnalysisPipeline:
    """
    Pipeline for segment analysis to evaluate the data
    model training
    """

    def __init__(self):
        pass

    def _print_distribution(self, y: pd.Series, split_name: str):
        """Prints distribution."""
        if y is None or len(y) == 0:
            print(f"   {split_name}: [EMPTY]")
            return

        total = len(y)
        counts = y.value_counts().sort_index()

        print(f"   {split_name} (Total: {total}):")
        for cls_val, count in counts.items():
            pct = (count / total) * 100
            print(f"     - Class {cls_val}: {count:>5} ({pct:5.1f}%)")

    def run_analysis(self, data: dict):
        """
        Logs analysis of data for a segment
        """
        seg_name = data['name']
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

        print(f"\n{'=' * 10} ANALYSIS: {seg_name} {'=' * 10}")

        n_train = len(X_train)
        n_test = len(X_test) if X_test is not None else 0
        total = n_train + n_test

        print(f"1. Data Volume:")
        print(f"   Total: {total} rows")
        if total > 0:
            print(f"   Split: Train {n_train} ({n_train / total:.1%}) | Test {n_test} ({n_test / total:.1%})")

        print(f"   Features: {X_train.shape[1]}")

        print(f"2. Target Distribution:")
        self._print_distribution(y_train, "Train Set")
        if y_test is not None:
            self._print_distribution(y_test, "Test Set ")

        if y_train.nunique() < 2:
            print(f"   [CRITICAL] Training impossible! Only 1 class found in Train.")

        print("-" * 40)


class TrainingPipeline:
    """
    Connects SegmentManager, Trainer, Evaluator и Deployer.
    """

    def __init__(self, trainer, evaluator, deployer):
        self.trainer = trainer
        self.evaluator = evaluator
        self.deployer = deployer

    def run_segment(self, data: dict) -> dict:
        """
        Executes the whole cycle for one segment
        """
        seg_name = data['name']
        X_train_raw, y_train = data['X_train'], data['y_train']
        X_test_raw, y_test = data['X_test'], data['y_test']

        current_grid = cfg.XGB_PARAM_GRID.copy()
        if data.get('override_grid'):
            print(f"   [CONFIG] Applying custom params for {seg_name}")
            current_grid.update(data['override_grid'])

        print(f"\n{'=' * 10} Processing: {seg_name} {'=' * 10}")

        # 1. train
        best_model = self.trainer.train(
            X_train_raw, y_train,
            param_grid=current_grid,
            scoring='f1'
        )
        self.deployer.save_model(best_model, seg_name)

        used_features = best_model.feature_names_in_
        X_train_clean = X_train_raw[used_features].copy()
        X_test_clean = None
        if X_test_raw is not None:
            X_test_clean = X_test_raw[used_features].copy()

        # 2. evaluate
        metrics = {}
        if X_test_clean is not None:
            y_pred = best_model.predict(X_test_clean)
            y_proba = best_model.predict_proba(X_test_clean)[:, 1]

            metrics = self.evaluator.calculate_metrics(y_test, y_pred, y_proba)
            self.deployer.save_metrics(metrics, seg_name)

            print(f"   Metrics: {metrics}")

        # 3. interpretation and plots
        self._run_interpretation(best_model, X_train_clean, X_test_clean, seg_name)

        return metrics

    def _run_interpretation(self, model, X_train, X_test, seg_name):
        """
        Builds and saves plots.
        """
        if X_test is None: return

        analyzer = ShapAnalyzer(model, X_train)
        print(f"   Generating plots for {seg_name}...")

        # A. Summary Plot
        analyzer.plot_summary(X_test, max_display=10)
        self.deployer.save_plot(f"SHAP_Summary_{seg_name}")
        plt.show()

        # B. Saturation Analysis (Top-3 features)
        top_df = analyzer.get_top_features(X_test, top_n=3)
        top_feats = top_df['Feature'].tolist()

        for feat in top_feats:
            if X_train[feat].nunique() > 2:
                analyzer.plot_dependence(X_test, feature=feat)
                self.deployer.save_plot(f"SHAP_Dependence_{seg_name}_{feat}")
                plt.show()

        # C. Delay Interaction
        if cfg.DELAY_OUTPUT_COL in X_test.columns:
            top_service = top_feats[0]
            analyzer.plot_interaction(
                X_test,
                feature_x=cfg.DELAY_OUTPUT_COL,
                feature_color=top_service
            )
            self.deployer.save_plot(f"SHAP_Interaction_{seg_name}")
            # plt.show()
            plt.close()