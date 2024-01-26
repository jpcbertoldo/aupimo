from dataclasses import dataclass, field

import matplotlib as mpl
import numpy as np


@dataclass
class RankDiagramDisplay:
    
    MODE_TRIANGLE = "triangle"
    MODE_PIRAMID = "piramid"
    MODES = (MODE_TRIANGLE, MODE_PIRAMID)

    average_ranks: np.ndarray = field(repr=False)
    methods_names: np.ndarray = field(repr=False)
    confidence_h1_matrix: np.ndarray = field(repr=False)
    min_confidence: float = 0.95
    
    # attributes from __post_init__()
    num_methods: int = field(init=False)
    
    # attributes from plot()
    ax_: mpl.axes.Axes = field(init=False, repr=False)
    fig_: mpl.figure.Figure = field(init=False, repr=False)
    
    def __post_init__(self):
        self.num_methods = len(self.methods_names)
        self.average_ranks = np.asarray(self.average_ranks)
        self.methods_names = np.asarray(self.methods_names)
        self.validate_args(self.average_ranks, self.methods_names, self.confidence_h1_matrix, self.min_confidence)
    
    @staticmethod
    def validate_args(average_ranks, methods_names, confidence_h1_matrix, min_confidence = 0.95, mode=None):
        
        num_methods = len(methods_names)
        
        assert average_ranks.shape == (num_methods,), f"{average_ranks.shape} != ({num_methods},)"
        assert ((average_ranks >= 1) & (average_ranks <= num_methods)).all(), "Invalid average ranks."        
        
        # make sure they are in increasing order
        assert np.all(np.diff(average_ranks) >= 0), "Invalid average ranks, must be in increasing order."
        
        assert all(methods_names), f"Invalid methods names: {methods_names}"
        assert (num_methods_names := len(methods_names)) == num_methods, f"Invalid methods names, expected {num_methods}, found {num_methods_names}"       
    
        assert confidence_h1_matrix.shape == (num_methods, num_methods), f"{confidence_h1_matrix.shape} != ({num_methods}, {num_methods})"
        # assert ((confidence_h1_matrix <= 1) & (confidence_h1_matrix >= 0)).all(), "Invalid confidence H1 matrix."        
        
        assert 0 <= min_confidence <= 1, f"Invalid `min_confidence` {min_confidence}"
        
        assert mode is None or mode in RankDiagramDisplay.MODES, f"Unkwown {mode=}, chose one of {RankDiagramDisplay.MODES}"
    
    def plot(self, ax, mode=MODE_TRIANGLE, **kwargs_functional):
        self.ax_ = ax
        self.fig_ = ax.figure        
        RankDiagramDisplay.plot_functional(
            ax, self.average_ranks, self.methods_names, self.confidence_h1_matrix, self.min_confidence, mode,
            **kwargs_functional,
        )
        
    @staticmethod
    def plot_functional(
        ax, average_ranks, methods_names, confidence_h1_matrix, min_confidence = 0.95, mode=MODE_TRIANGLE,
        stem_min_height = 0.5, stem_vspace = 0.25,
        low_confidence_position_low = 0.05, low_confidence_position_high = 0.30,
        stem_kwargs=None,
        bar_kwargs=None,
    ):
        
        RankDiagramDisplay.validate_args(average_ranks, methods_names, confidence_h1_matrix, min_confidence, mode)
        
        num_methods = len(methods_names)
        
        stem_heights = stem_min_height + stem_vspace * np.arange(0, num_methods) 

        if mode == RankDiagramDisplay.MODE_TRIANGLE:
            pass
                
        elif mode == RankDiagramDisplay.MODE_PIRAMID:
            right_maxidx = int(np.floor(num_methods / 2))
            left_minidx = num_methods - right_maxidx
            stem_heights[left_minidx:] = stem_heights[:right_maxidx][::-1]
        
        # stems (average ranks)
        ax.vlines(average_ranks, 0, stem_heights, linewidth=1, color="black")
        
        low_confidence_bar_position = 0

        # stems' annotations
        for idx, (name, rank, height,) in enumerate(zip(methods_names, average_ranks, stem_heights)):
            ax.annotate(
                f"{name} ({rank:.1f})", 
                xy=(rank, height), 
                textcoords="offset points", 
                va="baseline", 
                color="black", 
                linespacing=1,
                # manage the triangle vs. piramid config
                **(
                    dict(xytext=(4, 0), ha="left",)
                    if mode == RankDiagramDisplay.MODE_TRIANGLE or idx < right_maxidx else
                    dict(xytext=(-4, 0), ha="right",)
                ),
                **(stem_kwargs or {})
            )
            
            confidence_h1 = confidence_h1_matrix[(idx_prev := idx - 1), idx]
            
            if idx == 0 or confidence_h1 >= min_confidence: 
                low_confidence_bar_position = 0
                continue            
            
            rank_prev = average_ranks[idx_prev]
            low_confidence_bar_height = -(low_confidence_position_low if low_confidence_bar_position % 2 == 0 else low_confidence_position_high)
            ax.hlines(low_confidence_bar_height, rank, rank_prev, lw=3, color='k')
            ax.annotate(
                f"{conf_int:.0f}%" if (conf_int := int(100 * confidence_h1)) > 1 else "<1%",
                xy=((rank + rank_prev) / 2, low_confidence_bar_height),
                xytext=(0, 6),
                textcoords="offset points",
                va="baseline",
                ha="center",
                linespacing=1,
                **(bar_kwargs or {})
            )
            
            low_confidence_bar_position = not low_confidence_bar_position
        
        ax.set_aspect("equal")

        # X-axis
        ax.set_xlim(1, num_methods)
        ax.set_xscale('linear')
        ax.xaxis.set_inverted(True)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.tick_params(axis="x", which="major", direction="out", labeltop=False, labelbottom=True)
        
        # Y-axis
        ax.yaxis.set_inverted(True)
        ax.set_yscale('linear')
        ax.yaxis.set_visible(False)
        
        # onlu leave the bottom spine        
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.spines["bottom"].set_position("zero")

