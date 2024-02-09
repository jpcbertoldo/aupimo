"""Oracle functions for evaluating anomaly detection algorithms.

TODO: write docstring of `per_image/oracles.py`.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy import ndarray

from . import _validate, binclf_curve_numpy
from .binclf_curve_numpy import BinclfAlgorithm, per_image_binclf_curve, per_image_tfpr
from .pimo_numpy import _images_classes_from_masks

logger = logging.getLogger(__name__)


def get_per_image_threshs(
    anomaly_maps: ndarray,
    masks: ndarray,
    min_valid_score: float,
    num_threshs: int,
):
    """TODO write docstring of `per_image_threshs`.

    normal images have only nan values
    anomalous have a linearly spaced threshold sequence between min_valid_score
    and the maximum value of the anomaly map of the image (i.e. each image has a
    different sequence)
    """
    _validate.anomaly_maps(anomaly_maps)
    _validate.masks(masks)
    # TODO _validate.min_valid_score(min_valid_score)
    _validate.num_threshs(num_threshs)

    image_classes = _images_classes_from_masks(masks)

    per_image_threshs = []
    for anomaly_map, image_class in zip(anomaly_maps, image_classes):

        if image_class == 0 or (anomaly_map < min_valid_score).all():
            per_image_threshs.append(np.full(num_threshs, np.nan, dtype=anomaly_maps.dtype))
            continue

        in_asmap_max = anomaly_map.max()
        per_image_threshs.append(
            np.linspace(
                min_valid_score,
                in_asmap_max,
                num_threshs,
                dtype=anomaly_maps.dtype,
        ))
    return np.stack(per_image_threshs, axis=0)


def per_image_tfpr_curves(
    anomaly_maps: ndarray,
    masks: ndarray,
    min_valid_score: float,
    num_threshs: int,
    binclf_algorithm: str = BinclfAlgorithm.NUMBA,
) -> tuple[ndarray, ndarray]:
    """TODO write docstring of `per_image_tfpr_curves`.

    TODO: make it possible to compute the `min_valid_score` automatically
    it will require adding more args to manage the parameters...
    """
    _validate.anomaly_maps(anomaly_maps)
    _validate.masks(masks)
    # TODO _validate.min_valid_score(min_valid_score)
    _validate.num_threshs(num_threshs)
    BinclfAlgorithm.validate(binclf_algorithm)

    per_image_threshs = get_per_image_threshs(
        anomaly_maps, masks, min_valid_score, num_threshs
    )

    image_classes = _images_classes_from_masks(masks)

    _, per_image_binclf_anom = per_image_binclf_curve(
        anomaly_maps[image_classes == 1],
        masks[image_classes == 1],
        algorithm=binclf_algorithm,
        threshs_choice="given-per-image",
        threshs_given=per_image_threshs[image_classes == 1]
    )

    per_image_tfprs_anom = per_image_tfpr(per_image_binclf_anom)

    per_image_tfprs = np.full(
        (len(image_classes), num_threshs), np.nan, dtype=per_image_tfprs_anom.dtype
    )
    per_image_tfprs[image_classes == 1] = per_image_tfprs_anom

    return per_image_threshs, per_image_tfprs