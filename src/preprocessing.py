import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


class DataPreprocessor:
    """
    Class for data cleansing, generation new features and EDA.
    """

    def __init__(self):
        pass

    # ==========================================
    # 1. Info
    # ==========================================
    def get_metadata(self, df: pd.DataFrame):
        """
        Returns metadate: total list, numerical and categorical columns.
        """
        metadata = df.columns.tolist()
        numerical_cols = df.select_dtypes(include=["float64", "int64", "bool"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        return metadata, numerical_cols, categorical_cols

    # ==========================================
    # 2. Clean and Transform
    # ==========================================
    def drop_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Drops listed columns.
        """
        df = df.copy()
        existing_cols = [c for c in columns if c in df.columns]
        if existing_cols:
            df.drop(columns=existing_cols, inplace=True)
            print(f"Dropped columns: {existing_cols}")
        return df

    def fill_na(self, df: pd.DataFrame, columns: list, value=0) -> pd.DataFrame:
        """
        Fills NaN in listed columns with desired value.
        """
        df = df.copy()
        filled_count = 0

        for col in columns:
            if col in df.columns:
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna(value)
                    filled_count += 1

        if filled_count > 0:
            print(f"Filled NaNs with {value} in {filled_count} columns.")
        return df

    def replace_values(self, df: pd.DataFrame, columns: list, old_value, new_value) -> pd.DataFrame:
        """
        Replaces specific value for another in listed columns.
        """
        df = df.copy()
        replaced_cols = []

        for col in columns:
            if col in df.columns:
                # check if the value already exists
                if (df[col] == old_value).sum() > 0:
                    df[col] = df[col].replace(old_value, new_value)
                    replaced_cols.append(col)

        if replaced_cols:
            print(f"Replaced '{old_value}' with '{new_value}' in {len(replaced_cols)} columns.")

        return df

    def transform_log_sum(self, df: pd.DataFrame, input_cols: list, output_col: str,
                          drop_input: bool = True) -> pd.DataFrame:
        """
        Replaces or add column for log of their sum.
        """
        df = df.copy()

        valid_cols = [c for c in input_cols if c in df.columns]
        if not valid_cols:
            print(f"Warning: Input columns {input_cols} not found.")
            return df

        total_sum = df[valid_cols].sum(axis=1)

        # log transformation
        df[output_col] = np.log1p(total_sum)

        if drop_input:
            df.drop(columns=valid_cols, inplace=True)
            print(f"Log-transform applied to '{output_col}'. Dropped: {valid_cols}")

        return df

    # ==========================================
    # 3. Visualization (EDA)
    # ==========================================
    def plot_histograms(self, df: pd.DataFrame, numeric_cols: list, bins: int = 20):
        """Builds histograms for numerical features."""
        if not numeric_cols:
            return

        ncol_plots = 3
        nrow_plots = (len(numeric_cols) + ncol_plots - 1) // ncol_plots

        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 4 * nrow_plots))
        axs = axs.flatten()
        pastel_colors = sbn.color_palette("muted", len(numeric_cols))

        for i, col in enumerate(numeric_cols):
            sbn.histplot(df[col], color=pastel_colors[i], bins=bins, ax=axs[i])
            axs[i].set_title(f"Histogram: {col}")
            axs[i].set_xlabel(col)
            axs[i].set_ylabel("Frequency")

        # delete empty axes
        for j in range(len(numeric_cols), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    def plot_correlation(self, df: pd.DataFrame, cols: list = None):
        """Builds correlation matrix."""
        if cols:
            data_to_plot = df[cols]
        else:
            data_to_plot = df.select_dtypes(include='number')

        corr = data_to_plot.corr()

        plt.figure(figsize=(12, 10))
        sbn.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()

    def _autopct_large_only(self, pct, threshold):
        """ Help to cover small percentages in pie charts."""
        return f'{pct:.1f}%' if pct > threshold else ''

    def plot_piecharts(self, df: pd.DataFrame, categ_cols: list, threshold: float = 2.0):
        """
        Builds pie charts for categorical features.
        Small categories are in 'Other'
        """
        plt.close('all')

        ncol_plots = 2
        nrow_plots = (len(categ_cols) + ncol_plots - 1) // ncol_plots
        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 5 * nrow_plots))
        axs = axs.flatten()
        pastel_colors = sbn.color_palette("pastel")

        for i, col in enumerate(categ_cols):
            if col not in df.columns:
                continue

            results = df[col].value_counts()
            rel_freq = results / results.sum()

            # group small values in Other
            if threshold > 0:
                mask = rel_freq > threshold / 100
                other_sum = 1 - rel_freq[mask].sum()
                rel_freq = rel_freq[mask]
                if other_sum > 0.001:
                    rel_freq = pd.concat([rel_freq, pd.Series([other_sum], index=["Other"])])

            wedges, texts, autotexts = axs[i].pie(
                rel_freq.values,
                labels=rel_freq.index,
                autopct=lambda pct: self._autopct_large_only(pct, threshold),
                colors=pastel_colors[:len(rel_freq)],
                wedgeprops={'edgecolor': 'black'}
            )
            axs[i].set_title(f"Distribution: {col}", fontsize=14)

        for j in range(len(categ_cols), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    def plot_target_distribution(self, df: pd.DataFrame, target_col: str):
        """Build distribution of target column."""
        if target_col not in df.columns:
            return

        plt.figure(figsize=[8, 4])
        sbn.histplot(df[target_col], color='g', edgecolor="black", bins=3)
        plt.title(f"Target Variable Distribution: {target_col}")
        plt.show()