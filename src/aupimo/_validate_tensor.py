from numpy import ndarray
from torch import Tensor

from . import _validate


def is_tensor(tensor: Tensor, argname: str | None = None) -> None:
    """Validate that `tensor` is a `torch.Tensor`."""
    argname = f"'{argname}'" if argname is not None else "argument"

    if not isinstance(tensor, Tensor):
        msg = f"Expected {argname} to be a tensor, but got {type(tensor)}"
        raise TypeError(msg)


def safe_tensor_to_numpy(tensor: Tensor, argname: str | None = None) -> ndarray:
    """Convert a tensor to a numpy array, safely handling the device and dtype.

    TODO(jpcbertoldo): make sure this fucntion is used everywhere in the code.

    Args:
        tensor: The tensor to be converted.
        argname: The name of the argument, for error messages.

    Returns:
        The numpy array.
    """
    is_tensor(tensor, argname=argname)
    return tensor.detach().cpu().numpy()


def threshs(threshs: Tensor) -> None:
    _validate.threshs(safe_tensor_to_numpy(threshs, argname="threshs"))


def anomaly_maps(anomaly_maps: Tensor) -> None:
    _validate.anomaly_maps(safe_tensor_to_numpy(anomaly_maps, argname="anomaly_maps"))


def masks(masks: Tensor) -> None:
    _validate.masks(safe_tensor_to_numpy(masks, argname="masks"))
