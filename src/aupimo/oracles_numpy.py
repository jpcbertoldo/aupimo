"""Oracle functions for evaluating anomaly detection algorithms.

TODO: write docstring of module.
TODO(jpcbertoldo): test this module.
"""

import logging

from numpy import ndarray

from . import _validate
from .binclf_curve_numpy import BinclfAlgorithm, BinclfThreshsChoice, per_image_binclf_curve, per_image_iou

logger = logging.getLogger(__name__)


def per_image_iou_curves(
    anomaly_maps: ndarray,
    masks: ndarray,
    num_threshs: int,
    common_threshs: bool,
    binclf_algorithm: str = BinclfAlgorithm.NUMBA,
) -> tuple[ndarray, ndarray]:
    """TODO write docstring of `iou_curves`."""
    _validate.anomaly_maps(anomaly_maps)
    _validate.masks(masks)
    _validate.num_threshs(num_threshs)
    BinclfAlgorithm.validate(binclf_algorithm)

    threshs, binclfs = per_image_binclf_curve(
        anomaly_maps,
        masks,
        algorithm=binclf_algorithm,
        threshs_choice=(
            BinclfThreshsChoice.MINMAX_LINSPACE if common_threshs else BinclfThreshsChoice.MINMAX_LINSPACE_PER_IMAGE
        ),
    )

    ious = per_image_iou(binclfs)

    return threshs, ious
