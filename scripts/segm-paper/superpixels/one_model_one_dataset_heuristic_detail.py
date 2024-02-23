

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
num_levelsets = threshs.shape[0]
min_thresh = payload_loaded["min_thresh"]
upscale_factor = payload_loaded["upscale_factor"]

local_minima_idxs = payload_loaded["local_minima_idxs_per_image"][image_idx][:5]
local_minima_threshs = threshs[local_minima_idxs]

watershed_superpixel_relsize = payload_loaded["superpixels_params"]["superpixel_relsize"]
watershed_compactness = payload_loaded["superpixels_params"]["compactness"]

img = open_image(_convert_path(payload_loaded["paths"][image_idx]))
mask = safe_tensor_to_numpy(masks[image_idx])
asmap = safe_tensor_to_numpy(asmaps[image_idx])
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# resize image and asmap to double the resolution
img, asmap, mask = upscale_image_asmap_mask(img, asmap, mask, upscale_factor=upscale_factor)
valid_asmap, _ = valid_anomaly_score_maps(asmap[None, ...], min_thresh, return_mask=True)
valid_asmap = valid_asmap[0]
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# reproduce the call of where the two above come from
(superpixels, superpixels_boundaries_distance_map, _, __) = calculate_levelset_mean_dist_to_superpixel_boundaries_curve(
    img,
    asmap,
    min_thresh,
    watershed_superpixel_relsize,
    watershed_compactness,
    num_levelsets=num_levelsets,
)


def _get_cmap_transparent_bad(cmap_name: str = "jet"):
    cmap = mpl.cm.get_cmap(cmap_name)
    cmap.set_bad((0, 0, 0, 0))
    return cmap


def _get_binary_transparent_cmap(color) -> mpl.colors.ListedColormap:
    cmap = mpl.colors.ListedColormap([(0, 0, 0, 0), color, color, color])
    return cmap


fig, axes = plt.subplots(
    3,
    2,
    figsize=np.array((12, 18)) * 1.3,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
axrow = axes.flatten()

ax = axrow[0]
_ = plot_superpixels_boundaries_vs_gt(ax, img, superpixels, mask)

ax = axrow[1]
_ = ax.imshow(img)
_ = ax.imshow(valid_asmap, cmap=_get_cmap_transparent_bad("jet"), alpha=0.6)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=2.5,
    linestyles="--",
)

ax = axrow[2]
_ = ax.imshow(sk.segmentation.find_boundaries(superpixels, mode="outer"), cmap=_get_binary_transparent_cmap("magenta"))
_ = ax.imshow(valid_asmap, cmap=_get_cmap_transparent_bad("jet"), zorder=-10)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=2.5,
    linestyles="--",
)

ax = axrow[3]
_ = ax.imshow(superpixels_boundaries_distance_map)

ax = axrow[4]
_ = ax.imshow(img)
_ = ax.contour(
    asmap,
    levels=np.sort(local_minima_threshs[:5]),
    cmap="spring",
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=2.5,
    linestyles="--",
)

ax = axrow[5]
_ = ax.imshow(img)
_ = ax.contour(
    asmap,
    levels=[max_iou_per_image_result.threshs[image_idx].item()],
    colors=["red"],
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=2.5,
    linestyles="--",
)
