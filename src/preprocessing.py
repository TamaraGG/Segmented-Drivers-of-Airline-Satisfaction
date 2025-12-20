import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from typing import Callable

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

    def analyze_zeros(self, df: pd.DataFrame, columns: list, verbose: bool = True) -> pd.DataFrame:
        """
        Counts number and % of 0s in listed columns
        """

        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            print("No valid columns provided for zero analysis.")
            return pd.DataFrame()

        zero_counts = (df[valid_cols] == 0).sum()
        zero_percent = ((df[valid_cols] == 0).mean() * 100).round(2)

        analysis_df = pd.DataFrame({
            'Zeros Count': zero_counts,
            'Zeros %': zero_percent
        }).sort_values('Zeros %', ascending=False)

        zeros_present = analysis_df[analysis_df['Zeros Count'] > 0]

        if verbose:
            print("\n=== Zero Values Analysis ===")
            if not zeros_present.empty:
                print(zeros_present)
                print(f"\nTotal columns checked: {len(valid_cols)}")
            else:
                print("No zeros found in the specified columns.")

        return analysis_df

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

    def drop_rows_with_by_value(self, df: pd.DataFrame, columns: list, drop_value) -> pd.DataFrame:
        """
        Deletes rows with 0s.
        """
        df = df.copy()

        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            print("Warning: No valid columns provided for dropping zeros.")
            return df

        mask = (df[valid_cols] == drop_value).any(axis=1)

        rows_to_drop = mask.sum()

        if rows_to_drop > 0:
            original_len = len(df)
            df = df[~mask]
            print(
                f"Dropped {rows_to_drop} rows ({rows_to_drop / original_len:.2%}) containing {drop_value}s.")
        else:
            print(f"No {drop_value}s found in mandatory columns. No rows dropped.")

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

    def create_feature(self, df: pd.DataFrame, output_col: str,
                       func: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
        """
        Creates new column.

        :param df: original dataframe.
        :param output_col: name of new column.
        :param func: function that takes DF and returns Series.
        """
        df = df.copy()

        try:
            new_series = func(df)

            if len(new_series) != len(df):
                raise ValueError(f"Function returned length {len(new_series)}, expected {len(df)}")

            df[output_col] = new_series
            print(f"Feature created: '{output_col}'")

        except Exception as e:
            print(f"Error creating feature '{output_col}': {e}")
            raise e

        return df

    # ==========================================
    # 3. Visualization (EDA)
    # ==========================================
    def plot_histograms(self, df: pd.DataFrame, numeric_cols: list, bins: int = 20, log_y: bool = True):
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
            if log_y:
                axs[i].set_yscale('log')
                axs[i].set_ylabel("Frequency (Log)")
            else:
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

    def plot_interaction_evidence(self, df: pd.DataFrame, x_col: str, target_col: str, group_col: str):
        """
        Builds Interaction Plot
        Shows how x_col impacts on target_col according to different group_col.

        :param x_col: impact feature
        :param target_col: target
        :param group_col: segmentation feature
        """
        plt.figure(figsize=(10, 6))

        plot_df = df.copy()

        if plot_df[target_col].dtype == 'object':
            if 'satisfied' in plot_df[target_col].unique():
                temp_map = {'neutral or dissatisfied': 0, 'satisfied': 1}
                plot_df[target_col] = plot_df[target_col].map(temp_map)
            else:
                plot_df[target_col] = pd.factorize(plot_df[target_col])[0]

        sbn.lineplot(
            data=plot_df,
            x=x_col,
            y=target_col,
            hue=group_col,
            style=group_col,
            markers=True,
            dashes=False,
            linewidth=2.5,
            palette='deep',
            err_style='bars',
            errorbar=('ci', 95)
        )

        plt.title(f"Evidence for Segmentation: Interaction Effect\n({x_col} × {group_col})", fontsize=14)
        plt.ylabel(f"Probability of '{target_col}' (Satisfaction Rate)", fontsize=12)
        plt.xlabel(f"{x_col} Rating (1-5)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title=group_col, loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_baseline_divergence(self, df: pd.DataFrame, class_col: str, type_col: str, target_col: str):
        """
        Creates a grouped histogram showing differences in baseline happiness.
        Shows satisfaction levels for different combinations of travel classes and types.
        """
        plt.figure(figsize=(10, 6))

        plot_df = df.copy()

        if plot_df[target_col].dtype == 'object':
            if 'satisfied' in plot_df[target_col].unique():
                temp_map = {'neutral or dissatisfied': 0, 'satisfied': 1}
                plot_df[target_col] = plot_df[target_col].map(temp_map)
            else:
                plot_df[target_col] = pd.factorize(plot_df[target_col])[0]

        class_order = ['Eco', 'Eco Plus', 'Business']
        class_order = [c for c in class_order if c in plot_df[class_col].unique()]
        class_order = sorted(plot_df[class_col].unique())

        ax = sbn.barplot(
            data=plot_df,
            x=class_col,
            y=target_col,
            hue=type_col,
            order=class_order,
            palette='viridis',
            errorbar=None,
            edgecolor='black'
        )

        global_mean = plot_df[target_col].mean()
        plt.axhline(global_mean, color='red', linestyle='--', label=f'Global Average ({global_mean:.0%})')

        for container in ax.containers:
            labels = [f'{val:.0%}' for val in container.datavalues]
            ax.bar_label(container, labels=labels, padding=3, fontsize=11, fontweight='bold')

        plt.title(
            f"Evidence for Segmentation: Baseline Divergence\n(Satisfaction Rate by {type_col} & {class_col})",
            fontsize=14)
        plt.ylabel("Satisfaction Rate (%)", fontsize=12)
        plt.xlabel(class_col, fontsize=12)

        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

        plt.legend(title=type_col, loc='upper left')
        plt.tight_layout()
        plt.show()


    def plot_rank_shift(self, df: pd.DataFrame, service_cols: list, target_col: str, group_col: str,
                        top_n: int = 10, group_order: list = None):
        """
        Builds Slope Graph / Bump Chart.
        Shows how priority of factors differs between groups.

        :param service_cols: service columns.
        :param top_n: number of features to show.
        """
        df = df.copy()

        if df[target_col].dtype == 'object':
            if 'satisfied' in df[target_col].unique():
                df[target_col] = df[target_col].map({'neutral or dissatisfied': 0, 'satisfied': 1})
            else:
                df[target_col] = pd.factorize(df[target_col])[0]

        if group_order:
            groups = group_order
        else:
            groups = sorted(df[group_col].unique())

        if len(groups) < 2:
            print(f"Error: Need at least 2 groups for Rank Shift, found {len(groups)}")
            return

        rank_data = {}
        for g in groups:
            sub_df = df[df[group_col] == g]
            if sub_df.empty: continue
            corrs = sub_df[service_cols].corrwith(sub_df[target_col], method='spearman').abs()
            rank_data[g] = corrs.rank(ascending=False, method='min')

        rank_df = pd.DataFrame(rank_data)

        mask = (rank_df <= top_n).any(axis=1)
        plot_data = rank_df[mask].copy()

        fig, ax = plt.subplots(figsize=(12, 8))

        x_coords = range(len(groups))

        for feature in plot_data.index:
            y_ranks = plot_data.loc[feature, groups].values

            start_rank = y_ranks[0]
            end_rank = y_ranks[-1]

            if end_rank < start_rank:
                color = '#2ca02c'
            elif end_rank > start_rank:
                color = '#7f7f7f'
            else:
                color = '#1f77b4'

            ax.plot(x_coords, y_ranks, color=color, alpha=0.6, linewidth=2, marker='o')

            ax.text(x_coords[0] - 0.05, y_ranks[0], f"{feature} (#{int(y_ranks[0])})",
                    ha='right', va='center', fontsize=9, color='#333')

            ax.text(x_coords[-1] + 0.05, y_ranks[-1], f"(#{int(y_ranks[-1])}) {feature}",
                    ha='left', va='center', fontsize=9, color='#333')

            if len(groups) > 2:
                for ix, rank in zip(x_coords[1:-1], y_ranks[1:-1]):
                    ax.text(ix, rank - 0.2, int(rank), ha='center', va='bottom', fontsize=8, color=color)

        ax.set_xticks(x_coords)
        ax.set_xticklabels(groups, fontsize=12, fontweight='bold')

        ax.set_ylim(plot_data.max().max() + 1, 0)
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        for x in x_coords:
            ax.axvline(x, color='grey', alpha=0.1, zorder=0)

        plt.title(f"Rank Shift: Feature Importance Evolution\n({' -> '.join(groups)})", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_distribution_shift(self, df: pd.DataFrame, feature_col: str, group_col: str, hue_col: str = None):
        """
        Builds Boxplot for analysis of Distribution Shift.

        :param feature_col: numerical feature
        :param group_col: category 1
        :param hue_col: category 2
        """
        plt.figure(figsize=(10, 6))

        sbn.boxplot(
            data=df,
            x=group_col,
            y=feature_col,
            hue=hue_col,
            palette='Set2',
            linewidth=1.5,
            fliersize=2
        )

        if hue_col:
            medians = df.groupby([group_col, hue_col])[feature_col].median()

        plt.title(f"Evidence for Segmentation: Context Shift\n({feature_col} distribution by {group_col})", fontsize=14)
        plt.ylabel(feature_col, fontsize=12)
        plt.xlabel(group_col, fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)

        if hue_col:
            plt.legend(title=hue_col, loc='upper right')

        plt.tight_layout()
        plt.show()