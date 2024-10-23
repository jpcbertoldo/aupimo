
from dataclasses import dataclass, field
import matplotlib as mpl
import numpy as np
import pandas as pd


@dataclass
class BoxplotDisplay:
    
    df_gb_model: pd.DataFrame = field(repr=False)
    average_scores: np.ndarray = field(repr=False)
    models_names: np.ndarray = field(repr=False)
    confidence_h1_matrix: np.ndarray = field(repr=False)
    min_confidence: float = 0.95
    
    # attributes from __post_init__()
    num_models: int = field(init=False)
    
    # attributes from plot()
    ax_: mpl.axes.Axes = field(init=False, repr=False)
    fig_: mpl.figure.Figure = field(init=False, repr=False)
    
    def __post_init__(self):
        self.num_models = len(self.models_names)
        self.average_scores = np.asarray(self.average_scores)
        self.models_names = np.asarray(self.models_names)
        self.validate_args(self.df_gb_model, self.average_scores, self.models_names, self.confidence_h1_matrix, self.min_confidence)

    @staticmethod
    def validate_args_nobars(df_gb_model, models_names):
        
        assert all(models_names), f"Invalid models names: {models_names}"
        models_in_list = tuple(models_names)  
        
        # models_in_df = tuple(df_gb_model.index.tolist())
        # assert models_in_df == models_in_list, f"Invalid models in df_gb_model and list of models_names (must be in the same order): {models_in_df} != {models_in_list}"
        
        score_vals = df_gb_model.explode().values
        assert ((score_vals >= 0) & (score_vals <= 1)).all(), "Invalid df_gb_model."  

    @staticmethod
    def validate_args(df_gb_model, average_scores, models_names, confidence_h1_matrix, min_confidence = 0.95):

        BoxplotDisplay.validate_args_nobars(df_gb_model, models_names)
        
        num_models = len(models_names)

        assert average_scores.shape == (num_models,), f"{average_scores.shape} != ({num_models},)"
        assert ((average_scores >= 0) & (average_scores <= 1)).all(), "Invalid average scores."        
        
        assert confidence_h1_matrix.shape == (num_models, num_models), f"{confidence_h1_matrix.shape} != ({num_models}, {num_models})"
        assert ((confidence_h1_matrix <= 1) & (confidence_h1_matrix >= 0)).all(), "Invalid confidence H1 matrix."        
        
        assert 0 <= min_confidence <= 1, f"Invalid `min_confidence` {min_confidence}"
    
    def plot(self, ax, vert=True):
        self.ax_ = ax
        self.fig_ = ax.figure   
        if vert:
            BoxplotDisplay.plot_functional(ax, self.df_gb_model, self.average_scores, self.models_names, self.confidence_h1_matrix, self.min_confidence)
        else:
            BoxplotDisplay.plot_horizontal_functional(ax, self.df_gb_model, self.models_names)
        
    @staticmethod
    def plot_functional(ax, df_gb_model, average_scores, models_names, confidence_h1_matrix, min_confidence = 0.95):
        
        BoxplotDisplay.validate_args(df_gb_model, average_scores, models_names, confidence_h1_matrix, min_confidence)
        
        ax.boxplot(
            df_gb_model, 
            labels=models_names, 
            showmeans=True,
            medianprops={"color": "black"},
            meanprops={"marker": "o", "markeredgecolor": "black", "markerfacecolor": "black"},
            flierprops={"marker": "x", "markeredgecolor": "black", "markerfacecolor": "black"},
            widths=0.7,
            vert=True,
        )
        
        # plot confidence intervals for adjacent models
        for idx in range(len(models_names) - 1):

            confidence_h1 = confidence_h1_matrix[idx, (idx_adj := idx + 1)]
            if confidence_h1 >= min_confidence: continue
            
            ax.hlines(
                y=0,
                xmin=(x := idx + 1) + (x_eps := 0.10),
                xmax=(x_adj := idx_adj + 1) - x_eps,
                color="black",
                linestyle="-",
                linewidth=5,
            )
            ax.annotate(
                f"{conf_int:.0f}%" if (conf_int := int(100 * confidence_h1)) > 1 else "<1%",
                xy=((x + x_adj) / 2, 0),
                xycoords="data",
                xytext=(0, 6),
                textcoords="offset points",
                va="baseline",
                ha="center",
                color="black",
                fontsize=14,
                linespacing=1,
            )
        
        ax.set_xlabel("Model", fontsize=16)
        ax.grid(axis="x")
        ax.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.5)
        ax.xaxis.set_tick_params(rotation=30, length=15, labelsize=13)
        ax.xaxis.set_inverted(True)

        ax.set_ylabel("Score", fontsize=16)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y")
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")
        ax.yaxis.set_tick_params(length=15, labelsize=14)
    
    @staticmethod
    def plot_horizontal_functional(
        ax, df_gb_model, models_names,
        medianprops=None, meanprops=None, flierprops=None, widths=None, 
        boxprops=None, whiskerprops=None, capprops=None,
    ):
        BoxplotDisplay.validate_args_nobars(df_gb_model, models_names)
        
        ax.boxplot(
            df_gb_model, 
            labels=models_names, 
            showmeans=True,
            medianprops={
                "color": "black", 
                **(medianprops or {})
            },
            meanprops={
                "marker": "d", "markeredgecolor": "black", "markerfacecolor": "white", 
                **(meanprops or {})
            },
            flierprops={
                "marker": ".", "markeredgecolor": "black", "markerfacecolor": "black", "alpha": 0.25,
                "zorder": 10, 
                **(flierprops or {})
            },
            boxprops={
                "color": "black",
                **(boxprops or {})
            },
            whiskerprops={**(whiskerprops or {})},
            capprops={**(capprops or {})},
            widths=widths or 0.5,
            vert=False,
            zorder=10,
        )

        # Y-axis (Model)
        ax.yaxis.set_inverted(True)
        ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.3, zorder=-10)

        # X-axis (AUPIMO)
        ax.set_xlabel("AUPIMO")
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%",)
