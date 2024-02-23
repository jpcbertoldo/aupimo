

# %%

from aupimo.oracles_numpy import (
    open_image,
    upscale_image_asmap_mask,
)

ioucurves = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_local_threshs.pt")
max_iou_per_image_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

payload_loaded = torch.load(superpixel_bound_dist_heuristic_dir / "superpixel_bound_dist_heuristic.pt")

image_idx = 6

threshs = payload_loaded["threshs_per_image"][image_idx]
heuristic_curve_values = payload_loaded["levelset_mean_dist_curve_per_image"][image_idx]
num_levelsets = threshs.shape[0]
min_thresh = payload_loaded["min_thresh"]
upscale_factor = payload_loaded["upscale_factor"]

local_minima_idxs = payload_loaded["local_minima_idxs_per_image"][image_idx][:5]
local_minima_threshs = threshs[local_minima_idxs]
local_minima_values = heuristic_curve_values[local_minima_idxs]

img = open_image(_convert_path(payload_loaded["paths"][image_idx]))
mask = safe_tensor_to_numpy(masks[image_idx])
asmap = safe_tensor_to_numpy(asmaps[image_idx])
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# resize image and asmap to double the resolution
img, asmap, mask = upscale_image_asmap_mask(img, asmap, mask, upscale_factor=upscale_factor)
valid_asmap, _ = valid_anomaly_score_maps(asmap[None, ...], min_thresh, return_mask=True)
valid_asmap = valid_asmap[0]
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


fig, ax = plt.subplots()
_ = ax.plot(
    threshs,
    heuristic_curve_values,
    color="blue",
    label="levelset mean dist",
)
_ = ax.scatter(local_minima_threshs, local_minima_values, color="red")
_ = ax.plot(ioucurves.threshs[image_idx], ioucurves.per_image_ious[image_idx], color="green", label="iou")
_ = ax.scatter(
    local_minima_threshs,
    ioucurves.per_image_ious[image_idx][np.searchsorted(ioucurves.threshs[image_idx], local_minima_threshs)],
    color="red",
)
_ = ax.axvline(max_iou_per_image_result.threshs[image_idx], color="orange", label="max iou", linestyle="--")
_ = ax.axvline(min_thresh, color="black", label="min thresh", linestyle="--")


def _get_cmap_transparent_bad(cmap_name: str):
    cmap = mpl.cm.get_cmap(cmap_name)
    cmap.set_bad((0, 0, 0, 0))
    return cmap


fig, axes = plt.subplots(
    1,
    2,
    figsize=np.array((12, 6)) * 1,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
axrow = axes.flatten()

ax = axrow[0]
_ = ax.imshow(img)
_ = ax.imshow(valid_asmap, cmap=_get_cmap_transparent_bad("jet"), alpha=0.6)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=2.5,
    linestyles="--",
)
_ = ax.contour(
    asmap,
    levels=[max_iou_per_image_result.threshs[image_idx].item()],
    colors=["white"],
    linewidths=2.5,
)

ax = axrow[1]
_ = ax.imshow(img)
_ = ax.contour(
    asmap,
    levels=np.sort(local_minima_threshs),
    linewidths=2.5,
    cmap="spring",
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=2.5,
    linestyles="--",
)
