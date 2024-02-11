"""AUPIMO: Area Under the Per-IMage Overlap curve."""

from .binclf_curve import per_image_binclf_curve, per_image_fpr, per_image_iou, per_image_prec, per_image_tpr
from .binclf_curve_numpy import BinclfAlgorithm, BinclfThreshsChoice
from .oracles import IOUCurvesResult, per_image_iou_curves
from .pimo import AUPIMO, PIMO, AUPIMOResult, PIMOResult, aupimo_scores, pimo_curves
from .pimo_numpy import PIMOSharedFPRMetric
from .utils import (
    compare_models_pairwise_ttest_rel,
    compare_models_pairwise_wilcoxon,
    format_pairwise_tests_results,
    per_image_scores_stats,
)
from .utils_numpy import StatsOutliersPolicy, StatsRepeatedPolicy

__all__ = [
    # constants
    "BinclfAlgorithm",
    "BinclfThreshsChoice",
    "StatsOutliersPolicy",
    "StatsRepeatedPolicy",
    "PIMOSharedFPRMetric",
    # result classes
    "PIMOResult",
    "AUPIMOResult",
    "IOUCurvesResult",
    # functional interfaces
    "per_image_binclf_curve",
    "per_image_fpr",
    "per_image_tpr",
    "per_image_prec",
    "per_image_iou",
    "pimo_curves",
    "aupimo_scores",
    "per_image_iou_curves",
    # torchmetrics interfaces
    "PIMO",
    "AUPIMO",
    # utils
    "compare_models_pairwise_ttest_rel",
    "compare_models_pairwise_wilcoxon",
    "format_pairwise_tests_results",
    "per_image_scores_stats",
]
