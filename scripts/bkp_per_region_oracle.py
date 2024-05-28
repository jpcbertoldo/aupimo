# %%


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk

from aupimo._validate_tensor import safe_tensor_to_numpy
from aupimo.oracles_numpy import (
    get_superpixels_watershed,
    plot_superpixels_boundaries_vs_gt,
    plot_superpixels_colored_vs_gt,
)

min_thresh = _get_aupimo_thresh_lower_bound()

preds = []
thresh_per_valid_region_per_image = []
ious_optimal_threhs_per_valid_region = []

for image_idx, (asmap, mask) in enumerate(zip(safe_tensor_to_numpy(asmaps), safe_tensor_to_numpy(masks), strict=True)):
    valid_asmap_mask = asmap > min_thresh
    valid_asmap_regions = sk.measure.label(valid_asmap_mask, connectivity=1, background=0)
    valid_asmap_regions_labels = sorted(set(map(int, np.unique(valid_asmap_regions))) - {0})

    gt_regions = sk.measure.label(mask, connectivity=1, background=0)
    gt_regions_labels = sorted(set(map(int, np.unique(gt_regions))) - {0})

    pred = np.zeros_like(mask, dtype=bool)
    thresh_per_valid_region = {}

    for label in valid_asmap_regions_labels:
        valid_asmap_region_mask = valid_asmap_regions == label
        asmap_valid_region = asmap.copy()
        asmap_valid_region[~valid_asmap_region_mask] = min_thresh - 1

        gt_regions_labels_touched_by_region = sorted(set(gt_regions[valid_asmap_region_mask]) - {0})

        if len(gt_regions_labels_touched_by_region) == 0:
            gt_regions_touched_by_region = np.zeros_like(gt_regions, dtype=bool)
        else:
            gt_regions_touched_by_region = np.isin(gt_regions, gt_regions_labels_touched_by_region)

        iou_curve = per_image_iou_curves(
            torch.from_numpy(asmap_valid_region).unsqueeze(0),
            torch.from_numpy(gt_regions_touched_by_region).to(torch.bool).unsqueeze(0),
            num_threshs=3_000,
            common_threshs=False,
        )

        optimal_thresh_in_region = (
            max_iou_per_image(
                iou_curve.threshs,
                iou_curve.per_image_ious,
                iou_curve.image_classes,
                min_thresh=min_thresh,
            )
            .threshs[0]
            .item()
        )

        thresh_per_valid_region[label] = optimal_thresh_in_region

        # it can be `nan` for normal images, which here corresponds to
        # valid regions not touching the ground truth
        if np.isnan(optimal_thresh_in_region):
            continue

        pred |= asmap_valid_region >= optimal_thresh_in_region

    preds.append(pred)
    thresh_per_valid_region_per_image.append(thresh_per_valid_region)

    iou = (pred & mask).sum() / (pred | mask).sum() if mask.sum() > 0 else np.nan
    ious_optimal_threhs_per_valid_region.append(float(iou))

ious_optimal_threhs_per_valid_region = np.array(ious_optimal_threhs_per_valid_region)
preds = np.stack(preds, axis=0)

# %%

from aupimo.oracles import MaxIOUPerImageResult

per_image_thresh = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

# %%
# boxplots of ious from `preds_ious` vs `ious_maxs_result.ious`
fig, ax = plt.subplots()
_ = ax.boxplot(
    [
        per_image_thresh.ious[per_image_thresh.image_classes == 1],
        ious_optimal_threhs_per_valid_region[per_image_thresh.image_classes == 1],
    ],
    vert=False,
    labels=["1 per image", "1 per valid region"],
)

# %%
image_idx = 19
asmap = asmaps[image_idx]
mask = masks[image_idx]
if (img := plt.imread(images_abspaths[image_idx])).ndim == 2:
    img = img[..., None].repeat(3, axis=-1)

asmap = safe_tensor_to_numpy(asmap)
mask = safe_tensor_to_numpy(mask)

valid_asmap_mask = asmap > _get_aupimo_thresh_lower_bound()
valid_asmap = asmap.copy()
valid_asmap[~valid_asmap_mask] = np.nan

valid_asmap_regions = sk.measure.label(valid_asmap_mask, connectivity=1, background=0)
valid_asmap_regions_labels = sorted(set(map(int, np.unique(valid_asmap_regions))) - {0})

suppixs = get_superpixels_watershed(
    img,
    superpixel_relsize=1e-4,
    compactness=1e-4,
)
valid_suppixs_lbls = sorted(set(map(int, np.unique(suppixs * valid_asmap_mask))) - {0})
valid_suppixs_mask = np.isin(suppixs, list(valid_suppixs_lbls))
valid_suppixs = suppixs * valid_suppixs_mask
print(f"number of superpixels valid: {len(valid_suppixs_lbls)}")

fig, axes = plt.subplots(
    2,
    2,
    figsize=np.array((12, 12)) * 1.0,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
axrow = axes.flatten()

ts = np.linspace(0, 1, 40)
ts = np.concatenate(list(zip((ts_even := ts[0::2]), (ts_odd := ts[1::2][::-1]))))  # noqa: B905
suppixs_colors = list(map(tuple, mpl.colormaps["jet"](ts)))

ax = axrow[0]
_ = plot_superpixels_boundaries_vs_gt(ax, img, valid_suppixs, mask)

ax = axrow[1]
_ = plot_superpixels_colored_vs_gt(ax, valid_suppixs, mask)

ax = axrow[2]
cmap = mpl.cm.get_cmap("jet")
cmap.set_bad((0, 0, 0, 0))
_ = ax.imshow(valid_asmap, cmap=cmap)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=2.5,
    linestyles="--",
)

ax = axrow[3]
_ = ax.contour(
    preds[image_idx],
    levels=[0.5],
    colors="purple",
    linewidths=4.5,
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=2.5,
    linestyles="--",
)
single_thresh = (
    MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json").threshs[image_idx].item()
)
print(f"single_thresh: {single_thresh=}")
_ = ax.contour(
    asmap >= single_thresh,
    levels=[0.5],
    colors="red",
)

# %%

