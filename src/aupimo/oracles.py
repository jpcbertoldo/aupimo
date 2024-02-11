"""Oracle functions for evaluating anomaly detection algorithms.

This module implements torch interfaces to access the numpy code in `pimo_numpy.py`.
Check its docstring for more details.

Validations will preferably happen in ndarray so the numpy code can be reused without torch,
so often times the Tensor arguments will be converted to ndarray and then validated.

TODO(jpcbertoldo): test this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from aupimo import _validate, _validate_tensor

from . import oracles_numpy
from .binclf_curve_numpy import BinclfAlgorithm

if TYPE_CHECKING:
    from pathlib import Path


def _validate_per_image_ious(per_image_ious: Tensor, image_classes: Tensor) -> None:
    _validate.images_classes(_validate_tensor.safe_tensor_to_numpy(image_classes, argname="image_classes"))

    per_image_ious_array = _validate_tensor.safe_tensor_to_numpy(per_image_ious, argname="per_image_ious")

    # general validations
    _validate.per_image_rate_curves(
        per_image_ious_array,
        nan_allowed=True,  # normal images have NaNs
        decreasing=None,  # not applicable
    )

    # specific to anomalous images
    _validate.per_image_rate_curves(
        per_image_ious_array[image_classes == 1],
        nan_allowed=False,
        decreasing=None,  # not applicable
    )

    # specific to normal images
    normal_images_ious = per_image_ious[image_classes == 0]
    if not normal_images_ious.isnan().all():
        msg = "Expected all normal images to have NaN IoUs, but some have non-NaN values."
        raise ValueError(msg)


@dataclass
class IOUCurvesResult:
    """TODO(jpcbertoldo): doc IOUCurves.

    Notation:
        - N: number of images
        - K: number of thresholds
        - FPR: False Positive Rate
        - TPR: True Positive Rate
    """

    # metadata
    common_threshs: bool  # whether the thresholds are common to all images or per image

    # data
    threshs: Tensor = field(repr=False)  # shape => (K,) or (N, K)
    per_image_ious: Tensor = field(repr=False)  # shape => (N, K)

    # optional metadata
    paths: list[str] | None = field(repr=False, default=None)

    @property
    def num_threshs(self) -> int:
        """Number of thresholds."""
        return self.threshs.shape[-1]

    @property
    def num_images(self) -> int:
        """Number of images."""
        return self.per_image_ious.shape[0]

    @property
    def image_classes(self) -> Tensor:
        """Image classes (0: normal, 1: anomalous).

        Deduced from IOU values.
        If IOU values are not NaN, the image is considered anomalous.
        """
        return (~torch.isnan(self.per_image_ious)).any(dim=1).to(torch.int32)

    def __post_init__(self) -> None:
        """Validate the inputs for the result object are consistent."""
        try:
            if self.common_threshs:
                _validate_tensor.threshs(self.threshs)
            else:
                _validate_tensor.threshs_per_instance(self.threshs)
            _validate_per_image_ious(self.per_image_ious, self.image_classes)

            if self.paths is not None:
                _validate.source_images_paths(self.paths, expected_num_paths=self.per_image_ious.shape[0])

        except (TypeError, ValueError) as ex:
            msg = f"Invalid inputs for {self.__class__.__name__} object. Cause: {ex}."
            raise TypeError(msg) from ex

        if self.common_threshs and self.threshs.shape[0] != self.per_image_ious.shape[1]:
            msg = (
                f"Invalid {self.__class__.__name__} object. Attributes have inconsistent shapes: "
                f"{self.threshs.shape[0]=} != {self.per_image_ious.shape[1]=}."
            )
            raise TypeError(msg)

        if not self.common_threshs and self.threshs.shape != self.per_image_ious.shape:
            msg = (
                f"Invalid {self.__class__.__name__} object. Attributes have inconsistent shapes: "
                f"{self.threshs.shape=} != {self.per_image_ious.shape=}."
            )
            raise TypeError(msg)

    def to_dict(self) -> dict[str, Tensor | str]:
        """Return a dictionary with the result object's attributes."""
        dic = {"threshs": self.threshs, "per_image_ious": self.per_image_ious, "common_threshs": self.common_threshs}
        if self.paths is not None:
            dic["paths"] = self.paths
        return dic

    @classmethod
    def from_dict(cls: type[IOUCurvesResult], dic: dict[str, Tensor | str | list[str]]) -> IOUCurvesResult:
        """Return a result object from a dictionary."""
        try:
            return cls(**dic)  # type: ignore[arg-type]

        except TypeError as ex:
            msg = f"Invalid input dictionary for {cls.__name__} object. Cause: {ex}."
            raise TypeError(msg) from ex

    def save(self, file_path: str | Path) -> None:
        """Save to a `.pt` file.

        Args:
            file_path: path to the `.pt` file where to save the curves to.
        """
        _validate.file_path(file_path, must_exist=False, extension=".pt", pathlib_ok=True)
        payload = self.to_dict()
        torch.save(payload, file_path)

    @classmethod
    def load(cls: type[IOUCurvesResult], file_path: str | Path) -> IOUCurvesResult:
        """Load from a `.pt` file.

        Args:
            file_path: path to the `.pt` file where to load the curves from.
        """
        _validate.file_path(file_path, must_exist=True, extension=".pt", pathlib_ok=True)
        payload = torch.load(file_path)
        if not isinstance(payload, dict):
            msg = f"Invalid content in file {file_path}. Must be a dictionary."
            raise TypeError(msg)
        try:
            return cls.from_dict(payload)
        except TypeError as ex:
            msg = f"Invalid content in file {file_path}. Cause: {ex}."
            raise TypeError(msg) from ex


def per_image_iou_curves(
    anomaly_maps: Tensor,
    masks: Tensor,
    num_threshs: int,
    common_threshs: bool,
    binclf_algorithm: str = BinclfAlgorithm.NUMBA,
    paths: list[str] | None = None,
) -> IOUCurvesResult:
    """TODO write docstring of `per_image_iou_curves`.

    TODO(jpcbertoldo): make it possible to compute the `min_valid_score` automatically
    it will require adding more args to manage the parameters...
    """
    anomaly_maps_array = _validate_tensor.safe_tensor_to_numpy(anomaly_maps, argname="anomaly_maps")
    masks_array = _validate_tensor.safe_tensor_to_numpy(masks, argname="masks")
    # validations happen in the numpy code

    if paths is not None:
        _validate.source_images_paths(paths, expected_num_paths=anomaly_maps_array.shape[0])

    threshs_array, ious_array = oracles_numpy.per_image_iou_curves(
        anomaly_maps_array,
        masks_array,
        num_threshs,
        binclf_algorithm=binclf_algorithm,
        common_threshs=common_threshs,
    )

    device = anomaly_maps.device
    # the shape is (K,) or (N, K)
    threshs = torch.from_numpy(threshs_array).to(device)
    # the shape is (N, K)
    ious = torch.from_numpy(ious_array).to(device)

    return IOUCurvesResult(common_threshs=common_threshs, threshs=threshs, per_image_ious=ious, paths=paths)
