
# %%
# asmap -> valid asmap

from pathlib import Path

from matplotlib import pyplot as plt
from torch import Tensor

from aupimo import AUPIMOResult
from aupimo.utils import valid_anomaly_score_maps

savedir: Path
asmaps: Tensor

def _get_aupimo_thresh_upper_bound(aupimoresult_fpath: Path) -> float:
    aupimoresult = AUPIMOResult.load(aupimoresult_fpath)
    return aupimoresult.thresh_bounds[1]


aupimoresult_fpath = savedir / "aupimo" / "aupimos.json"
MIN_VALID_THRESHOLD = _get_aupimo_thresh_upper_bound(aupimoresult_fpath)

vasmaps = valid_anomaly_score_maps(asmaps, MIN_VALID_THRESHOLD)


def _get_valid_cmap(cmap: str, bad: str = "gray") -> plt.cm.ScalarMappable:
    cmap = plt.cm.get_cmap(cmap)
    cmap.set_bad(bad)
    return cmap


ASMAP_CMAP = "jet"
VASMAP_CMAP = _get_valid_cmap("jet")

# %%
# viz: img vs asmap vs valid asmap

def _plot_gt_contour(ax, mask, **kwargs):
    kwargs = {**dict(colors="magenta", linewidths=(lw := 0.8)), **kwargs}
    return ax.contour(mask, levels=[0.5], **kwargs)

fig, axrow = plt.subplots(
    1, 3, figsize=(8, 4), 
    sharex=True, sharey=True, constrained_layout=True,
)
ax = axrow[0]
_ = ax.imshow(img)
_ = _plot_gt_contour(ax, mask)
_ = ax.set_title("image")

ax = axrow[1]
_ = ax.imshow(asmap, cmap=ASMAP_CMAP)
_ = _plot_gt_contour(ax, mask)
_ = ax.set_title("asmap (local colormap)")

ax = axrow[2]
_ = ax.imshow(vasmap, cmap=VASMAP_CMAP)
_ = _plot_gt_contour(ax, mask)
_ = ax.set_title("valid asmap (local colormap)")

for ax in axrow:
    _ = ax.axis("off")



# %%
# plot iou curves

from scipy.signal import argrelextrema

# find local maxima
iou_local_maxima_idxs = np.array(argrelextrema(iou_curves[:, 1].numpy(), np.greater, order=10, axis=1))

# mask to select thresholds as iou values from iou curves where there is a local maxima
iou_local_maxima_mask = np.zeros(iou_curves[:, 1].shape, dtype=bool)
_flat_mask = np.ravel_multi_index(iou_local_maxima_idxs, iou_local_maxima_mask.shape)
np.ravel(iou_local_maxima_mask)[_flat_mask] = True

# convert to torch
iou_local_maxima_idxs = torch.from_numpy(iou_local_maxima_idxs)
iou_local_maxima_mask = torch.from_numpy(iou_local_maxima_mask)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# by doing this, i am losing the information of "from which image" the local maxima is
# see the dicts in the cell below
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
iou_local_maxima = iou_curves[:, 1][iou_local_maxima_mask]
iou_local_maxima_threshs = iou_curves[:, 0][iou_local_maxima_mask]

iou_global_maxima_idx = torch.argmax(iou_curves[:, 1], dim=1)
iou_global_maxima = iou_curves[:, 1][torch.arange(len(iou_curves)), iou_global_maxima_idx]
iou_global_maxima_threshs = iou_curves[:, 0][torch.arange(len(iou_curves)), iou_global_maxima_idx]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
_ = ax.plot(iou_curves[:, 0].T, iou_curves[:, 1].T, label=np.arange(len(iou_curves)))
_ = ax.scatter(iou_local_maxima_threshs, iou_local_maxima, s=50, c="blue")
_ = ax.scatter(iou_global_maxima_threshs, iou_global_maxima, s=50, c="red")
_ = ax.set_xlabel("threshold")
_ = ax.set_ylabel("IoU")
_ = ax.set_ylim(0, 1)
_ = ax.legend(loc="upper right")

# %%
# viz max IoU contours

import numpy as np

image_idx = 6
asmap = asmaps[image_idx]
vasmap = vasmaps[image_idx]
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
if img.ndim == 2:
    img = img[..., None].repeat(3, axis=-1)

# get the global maximum
thresh = iou_global_maxima_threshs[image_idx]
thresh_idx = iou_global_maxima_idx[image_idx]

iou_of_thresh = iou_global_maxima[image_idx]

fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True)
axrow = axes.ravel()
_ = axrow[0].imshow(img)
_ = axrow[0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[1].imshow(asmap, cmap="gray")
_ = axrow[1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
cs = axrow[1].contour(asmap, [thresh], colors=["red"], linewidths=lw)
fmt = {l: f"{val:.0%}" for l, val in zip(cs.levels, [iou_of_thresh], strict=False)}
_ = axrow[1].clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)
# get gray cmap and make nan values black
cmap = plt.cm.get_cmap("gray")
_ = cmap.set_bad("gray")
_ = axrow[2].imshow(vasmap, cmap=cmap)
_ = axrow[2].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[2].contour(vasmap, [thresh], colors=["red"], linewidths=lw)

































# %%
# superpixel segmentation example
# src: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html#comparison-of-segmentation-and-superpixel-algorithms

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, mark_boundaries, quickshift, slic, watershed

image_idx = 5
asmap = asmaps[image_idx]
vasmap = _asmap2vasmap_aupimo_bounded(asmap)
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
threshs = tfpr_pivots_thresh[image_idx]

# using info from aupimo's x-axis bounds to determine `min_size`
aupimo_min_size = int(np.ceil(1e-5 * np.prod(img.shape[:2])))
# 5 here

segments_fz = felzenszwalb(
    img,
    # higher scale --> larger clusters
    # img, scale=100,  # from the example
    scale=200,
    sigma=0.5,
    min_size=50,  # from the example
    # too small (~5)
    # min_size=aupimo_min_size,
)

# K-means in the 5d space: Color-(x,y,z) + image location
segments_slic = slic(
    img,
    # nb of k-means clusters/centers
    n_segments=(n_segments:=2500),
    # tradeoff between color and space proximity; higher --> + weight to space
    compactness=10,
    # use LAB color space; useful?
    # convert2lab=True,
    # preprocessing blur smoothing in the 5d space, can have per-channel values
    sigma=0,  # deactivated
    # sigma=1,  # from the example
    # could it useful?
    # min_size_factor=aupimo_min_size / (supposed_size := np.prod(img.shape[:2]) / n_segments),
    # doc recommends 3; is it useful?
    # max_size_factor=3,
    start_label=1,
    # zero-parameter mode
    # slic_zero=True,
)

# mode-seeking algorithm (mean-shift) in color-(x, y) space
segments_quick = quickshift(
    img,
    # gaussian kernel used for smoothing the sample density
    kernel_size=3, 
    # preprocessing blur smoothing
    sigma=0,  # deactivated
    # cut off point for the neighborhood in the mean-shift procedure (not sure?)
    # higher --> less clusters
    max_dist=6,
    # color-space tradeoff; higher --> + weight to color
    ratio=.1,
    convert2lab=False,
    rng=42,
)

segments_watershed = watershed(
    (gradient := sobel(rgb2gray(img))), 
    markers=2500, 
    # makes it harder for markers to flood faraway pixels --> regions more regularly shaped
    compactness=0.001
)

segments_watershed_in_vasmap_mask = watershed(
    (gradient := sobel(rgb2gray(img))), 
    # markers=2500, 
    markers=(vasmap_mask := ~vasmap.isnan().numpy()).sum() / 100, 
    # makes it harder for markers to flood faraway pixels --> regions more regularly shaped
    # compactness=0.001,  # from the example
    compactness=3e-4,
    mask=vasmap_mask,
)

print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")
print(f"Watershed in vasmap mask number of segments: {len(np.unique(segments_watershed_in_vasmap_mask))}")

fig, ax = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True, constrained_layout=True)
ax = ax.T
_ = ax[0, 0].imshow(mark_boundaries(img, segments_fz))
_ = ax[0, 0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = ax[0, 0].set_title("Felzenszwalbs's method")
_ = ax[0, 1].imshow(mark_boundaries(img, segments_slic))
_ = ax[0, 1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = ax[0, 1].set_title("SLIC")
_ = ax[1, 0].imshow(mark_boundaries(img, segments_quick))
_ = ax[1, 0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = ax[1, 0].set_title("Quickshift")
_ = ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
_ = ax[1, 1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = ax[1, 1].set_title("Compact watershed")
_ = ax[2, 1].imshow(mark_boundaries(img, segments_watershed_in_vasmap_mask))
_ = ax[2, 1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = ax[2, 1].set_title("Compact watershed in vasmap mask")

# %%
# best achievable iou with superpixel segmentation (COMPACT WATERSHED)

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, mark_boundaries, quickshift, slic, watershed

image_idx = 5
asmap = asmaps[image_idx]
vasmap = _asmap2vasmap_aupimo_bounded(asmap)
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
threshs = tfpr_pivots_thresh[image_idx]

# resolution of the original image 
# each superpixel is a zone with an unique value, like a semantic segmentation
segments = watershed(
    (gradient := sobel(rgb2gray(img))), 
    # markers=2500, 
    markers=(vasmap_mask := ~vasmap.isnan().numpy()).sum() / 100, 
    # makes it harder for markers to flood faraway pixels --> regions more regularly shaped
    # compactness=0.001,  # from the example
    compactness=3e-4,
    mask=vasmap_mask,
)
segment_values = np.unique(segments[segments != 0])
num_segments = len(segment_values)
print(f"number of segments: {num_segments}")

# =============================================================================
# init segment selection

# find segments 100% inside the ground truth mask
# TODO speed up by converting to sets of pixel indexes and using set operations?
segments_inside_mask = {
    segment_value
    for segment_value in segment_values
    if mask[segments == segment_value].all()
}

# find segments 100% outside the ground truth mask
segments_outside_mask = np.array([
    segment_value
    for segment_value in segment_values
    if (~mask[segments == segment_value]).all()
])

selected_segments = set(segments_inside_mask)
available_segments = set(segment_values) - set(selected_segments) - set(segments_outside_mask)

print(f"{len(selected_segments)=} {len(available_segments)=}")

# =============================================================================
# viz segment selection (initial state)

import matplotlib as mpl
from numpy import ndarray

# zero means "out of scope", meaning it is not in the initial `segments`
segments_selection_viz = np.zeros_like(segments, dtype=int)

# 1 means "selected"
for segment_value in selected_segments:
    segments_selection_viz[segments == segment_value] = 1

# 2 means "available"
for segment_value in available_segments:
    segments_selection_viz[segments == segment_value] = 2
    
# 3 means "outside"
for segment_value in segments_outside_mask:
    segments_selection_viz[segments == segment_value] = 3
    
current_segmentation = segments_selection_viz == 1

def iou_of_segmentation(segmentation: ndarray, mask: ndarray) -> float:
    assert segmentation.shape == mask.shape, f"{segmentation.shape=} {mask.shape=}"
    assert segmentation.dtype == bool, f"{segmentation.dtype=}"
    assert mask.dtype == bool, f"{mask.dtype=}"
    return (segmentation & mask).sum() / (segmentation | mask).sum()

iou_of_selection = iou_of_segmentation(current_segmentation, mask.numpy())
avg_cont_dist_of_selection = compute_avg_contour_distance(current_segmentation, mask.numpy())

# define a custom colormap for the values above (0, 1, 2, 3)
colors = [(0, 0, 0, 0), "tab:green", "tab:blue", "tab:red"]
cmap = mpl.colors.ListedColormap(colors)
bounds = [0, 1, 2, 3, 4]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, axrow = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)

_ = axrow[0].imshow(mark_boundaries(img, segments))
_ = axrow[0].imshow(segments_selection_viz, cmap=cmap, norm=norm, alpha=0.5)
_ = axrow[0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[0].set_title("segment selection initial state")

csmask = axrow[1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8),)
cspred = axrow[1].contour(current_segmentation, levels=[0.5], colors="green", linewidths=(lw := 0.8),)
labels = [
    "ground truth mask",
    f"oracle segmentation\nIoU={iou_of_selection:.0%}\navg contour dist={avg_cont_dist_of_selection:.0%}",
]
for cs, label in zip([csmask, cspred], labels):
    cs.collections[0].set_label(label)
_ = axrow[1].legend()
_ = axrow[1].set_title("oracle segmentation")

_ = fig.suptitle("oracle segmentation (inital state)")

# =============================================================================
# find optimal segment selection to optimize iou
import copy

_current_segmentation = current_segmentation.copy()
_selected = copy.deepcopy(selected_segments)
_available = copy.deepcopy(available_segments)

history = [
    {
        "selected": copy.deepcopy(_selected),
        "iou": iou_of_segmentation(_current_segmentation, mask.numpy()),
    }
]

# best first search
while _available:
    # find the segment that maximizes the iou of the current segmentation
    best_segment = max(
        _available, 
        key=lambda segment_value: iou_of_segmentation(
            _current_segmentation | (segments == segment_value), 
            mask.numpy()
        )
    )
    _selected.add(best_segment)
    _available.remove(best_segment)
    _current_segmentation = _current_segmentation | (segments == best_segment)
    history.append({
        "best": best_segment,
        "selected": copy.deepcopy(_selected),
        "iou": iou_of_segmentation(_current_segmentation, mask.numpy()),
    })

# =============================================================================
# search history

iou_hist = np.array([h["iou"] for h in history])
best_iou_stepidx = np.argmax(iou_hist)

best_dict = history[best_iou_stepidx]
best_iou = best_dict["iou"]
best_selection = best_dict["selected"]

best_segmentation = np.zeros_like(segments, dtype=bool)
for segment_value in best_selection:
    best_segmentation[segments == segment_value] = True
    
iou_of_best = iou_of_segmentation(best_segmentation, mask.numpy())
avg_cont_dist_of_best = compute_avg_contour_distance(best_segmentation, mask.numpy())

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
_ = ax.plot(iou_hist)
_ = ax.scatter(best_iou_stepidx, best_iou, s=50, c='red')
_ = ax.set_xlabel("step index")
_ = ax.set_ylabel("IoU")
_ = ax.set_ylim(0, 1)
_ = ax.legend(loc="lower right")

fig, axrow = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)

_ = axrow[0].imshow(mark_boundaries(img, segments))
_ = axrow[0].imshow(segments_selection_viz, cmap=cmap, norm=norm, alpha=0.5)
_ = axrow[0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[0].contour(best_segmentation, levels=[0.5], colors="black", linewidths=(lw := 1.2))
_ = axrow[0].set_title("segment selection best iou")

csmask = axrow[1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8),)
cspred = axrow[1].contour(best_segmentation, levels=[0.5], colors="green", linewidths=(lw := 0.8),)
labels = [
    "ground truth mask",
    f"oracle segmentation\nIoU={iou_of_best:.0%}\navg contour dist={avg_cont_dist_of_best:.0%}",
]
for cs, label in zip([csmask, cspred], labels):
    cs.collections[0].set_label(label)
_ = axrow[1].legend()
_ = axrow[1].set_title("oracle segmentation")

_ = fig.suptitle("oracle segmentation (best iou)")
