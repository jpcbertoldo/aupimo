# %%
# generate superpixels

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk

image_idx = 6
asmap = asmaps[image_idx]
mask = masks[image_idx]
if (img := plt.imread(images_abspaths[image_idx])).ndim == 2:
    img = img[..., None].repeat(3, axis=-1)

num_pixels = img.shape[0] * img.shape[1]
makers_pixels_ratio = 0.01
num_markers = int(num_pixels * makers_pixels_ratio)
print(f"{num_pixels=} {makers_pixels_ratio=} {num_markers=}")

suppixs = sk.segmentation.watershed(
    (gradient := sk.filters.sobel(sk.color.rgb2gray(img))),
    markers=num_markers,
    # makes it harder for markers to flood faraway pixels --> regions more regularly shaped
    compactness=0.001,
)
suppixs_labels = sorted(set(np.unique(suppixs.flatten())) - {0})
print(f"number of superpixels: {(num_superpixels := len(set(np.unique(suppixs)) - {0}))}")

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
axrow = axes.flatten()

ts = np.linspace(0, 1, 40)
ts = np.concatenate(list(zip((ts_even := ts[0::2]), (ts_odd := ts[1::2][::-1]))))  # noqa: B905
suppixs_colors = list(map(tuple, mpl.colormaps["jet"](ts)))

ax = axrow[0]
_ = ax.imshow(
    sk.segmentation.mark_boundaries(img, suppixs, color=mpl.colors.to_rgb("magenta")),
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=(lw := 2.5),
    linestyles="--",
)

ax = axrow[1]
_ = ax.imshow(
    sk.color.label2rgb(suppixs, colors=suppixs_colors),
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=lw,
    linestyles="--",
)


# %%
# best achievable iou with superpixels

import copy

import matplotlib as mpl
from numpy import ndarray


def suppix_selection_map(
    suppixs: ndarray,
    selected_suppixs_labels: set[int],
    available_suppixs_labels: set[int],
) -> ndarray:
    # 0 means "out of scope", meaning it is not in the initial `segments`
    suppixs_selection_viz = np.full_like(suppixs, np.nan, dtype=float)
    # 1 means "selected"
    suppixs_selection_viz[np.isin(suppixs, list(selected_suppixs_labels))] = 1
    # 2 means "available"
    suppixs_selection_viz[np.isin(suppixs, list(available_suppixs_labels))] = 2
    return suppixs_selection_viz


def iou_of_segmentation(segmentation: ndarray, mask: ndarray) -> float:
    assert segmentation.shape == mask.shape, f"{segmentation.shape=} {mask.shape=}"
    assert segmentation.dtype == bool, f"{segmentation.dtype=}"
    assert mask.dtype == bool, f"{mask.dtype=}"
    return (segmentation & mask).sum() / (segmentation | mask).sum()


def get_initial_states_in_on_gt(suppixs: ndarray, mask: ndarray) -> tuple[set[int], set[int]]:
    """Return (initial selected, initial available)."""
    # superpixels touching the ground truth mask
    on_gt_labels = set(np.unique(suppixs * mask.astype(int))) - {0}

    # find segments 100% inside the ground truth mask
    in_gt_labels = sorted(
        {label for label in on_gt_labels if (mask[suppixs == label]).all()},
    )

    return set(in_gt_labels), set(on_gt_labels) - set(in_gt_labels)


def get_binclfs_per_suppix(suppixs: ndarray, mask: ndarray, labels: set[int] | None) -> ndarray:
    if labels is None:
        labels = sorted(set(np.unique(suppixs.flatten())) - {0})

    # +1 is for the background
    # the -1 values are dummy values (not used)
    binclf_per_suppix = -np.ones((num_superpixels + 1, 2, 2), dtype=int)

    for label in labels:
        suppix_mask = suppixs == label
        tp = (mask & suppix_mask).sum()
        fp = (~mask & suppix_mask).sum()
        tn = (~mask & ~suppix_mask).sum()
        fn = (mask & ~suppix_mask).sum()
        binclf_per_suppix[label, :] = np.array([[tn, fp], [fn, tp]])

    return binclf_per_suppix


def iou_of_selection(selected_labels: set[int], binclf_per_suppix: ndarray, mask: ndarray) -> float:
    binclfs = binclf_per_suppix[list(selected_labels)]
    tp = binclfs[:, 1, 1].sum()
    fp = binclfs[:, 0, 1].sum()
    gt_size = mask.sum()
    return tp / (fp + gt_size)


initial_selected_suppixs_labels, initial_available_suppixs_labels = get_initial_states_in_on_gt(suppixs, mask.numpy())


binclf_per_suppix = get_binclfs_per_suppix(
    suppixs,
    mask.numpy(),
    labels=initial_selected_suppixs_labels | initial_available_suppixs_labels,
)

selected_labels = copy.deepcopy(initial_selected_suppixs_labels)
available_labels = copy.deepcopy(initial_available_suppixs_labels)
current_segmentation = suppix_selection_map(suppixs, selected_labels, set()) == 1

history = [
    {
        "selected": copy.deepcopy(selected_labels),
        "iou": iou_of_segmentation(current_segmentation, mask.numpy()),
    },
]

# this can be much faster by computing each suppix's iou with the mask and sorting them
# best first search
for _ in range(len(initial_available_suppixs_labels)):
    # find the segment that maximizes the iou of the current segmentation
    chosen = max(
        available_labels,
        key=lambda candidate: iou_of_selection(
            selected_labels | {candidate},
            binclf_per_suppix,
            mask.numpy(),
        ),
    )
    current_segmentation |= suppixs == chosen
    iou = iou_of_segmentation(current_segmentation, mask.numpy())
    history.append(
        {
            "chosen": chosen,
            "iou": iou,
        },
    )
    # when the iou decreases, stop
    if history[-1]["iou"] < history[-2]["iou"]:
        # and remove the last element
        _ = history.pop()
        break
    selected_labels.add(chosen)
    available_labels.remove(chosen)


# =============================================================================
# search history

num_suppixs_hist = len(history[0]["selected"]) + np.arange(len(history))
iou_hist = np.array([h["iou"] for h in history])

# chosen history (intial step not in there)
chosen_hist = np.array([h["chosen"] for h in history[1:]])
chosen_candidate_mean_ascore_hist = np.array([asmap[suppixs == chosen].mean().item() for chosen in chosen_hist])

initial_iou = history[0]["iou"]
best_iou = history[-1]["iou"]
best_selection = history[0]["selected"] | {record["chosen"] for record in history[1:]}
best_segmentation = suppix_selection_map(suppixs, best_selection, set()) == 1

# -----------------------------------------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=np.array((4, 3)) * 2.0)

_ = ax.plot(num_suppixs_hist, iou_hist, label="iou history", marker="o", markersize=5, color="black")
_ = ax.axhline(best_iou, color="black", linestyle="--", label="best iou")
_ = ax.axhline(initial_iou, color="black", linestyle="--", label="initial iou")

twinax = ax.twinx()
_ = twinax.scatter(
    num_suppixs_hist[1:],
    chosen_candidate_mean_ascore_hist,
    label="superpixel mean ascore",
    marker="x",
    color="tab:blue",
)
_ = twinax.axhline(_get_aupimo_thresh_lower_bound(), color="tab:blue", linestyle="--", label="aupimo thresh")

_ = ax.set_xlabel("Number of superpixels selected")
# integer format
_ = ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

_ = ax.set_ylabel("IoU")
_ = ax.set_ylim(0, 1)
_ = ax.set_yticks(np.linspace(0, 1, 6))
_ = ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
_ = ax.grid(axis="y")

_ = twinax.set_ylabel("Superpixel anomaly score")
_ = twinax.set_yticks(np.linspace(*twinax.get_ylim(), 6))
_ = twinax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))


# =============================================================================
suppixs_selection_viz = suppix_selection_map(suppixs, initial_selected_suppixs_labels, initial_available_suppixs_labels)

# %%
# -----------------------------------------------------------------------------

# define a custom colormap for the values above (0, 1, 2, 3)
cmap = mpl.colors.ListedColormap([(0, 0, 0, 0), "tab:blue", "tab:olive"])
norm = mpl.colors.BoundaryNorm([0, 1, 2, 3], cmap.N)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
axrow = axes.flatten()

ax = axrow[0]
_ = ax.imshow(
    sk.segmentation.mark_boundaries(
        img,
        suppixs & ~np.isnan(suppixs_selection_viz),
        color=mpl.colors.to_rgb("magenta"),
    ),
)
_ = ax.imshow(suppixs_selection_viz, cmap=cmap, norm=norm, alpha=0.5)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=lw,
    linestyles="--",
)

ax = axrow[1]
_ = ax.imshow(img)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=lw,
    linestyles="--",
)
cs_gt = ax.contour(
    suppixs_selection_viz == 1,
    levels=[0.5],
    colors="magenta",
    linewidths=1,
    linestyles="-",
)

# %%
suppixs_mean_ascores = np.array([asmap[suppixs == label].mean().item() for label in sorted(suppixs_labels)])
suppixs_mean_ascores = np.array([asmap[suppixs == label].median().item() for label in sorted(suppixs_labels)])
suppixs_mean_ascores_argsort = np.argsort(suppixs_mean_ascores)[::-1]
suppixs_mean_ascores = suppixs_mean_ascores[suppixs_mean_ascores_argsort]
suppixs_labels_sorted = np.array(sorted(suppixs_labels))[suppixs_mean_ascores_argsort]
sorted_labels_selected_mask = np.isin(suppixs_labels_sorted, list(best_selection))

# %%
fig, axrow = plt.subplots(
    1,
    2,
    figsize=np.array((7, 3)) * 1.2,
    layout="constrained",
)
plot_idxs = np.arange(len(suppixs_mean_ascores))

ax = axrow[0]
_ = ax.scatter(
    plot_idxs[sorted_labels_selected_mask],
    suppixs_mean_ascores[sorted_labels_selected_mask],
    marker=".",
    color="tab:blue",
    label="selected",
    s=2,
    alpha=.3,
)
_ = ax.scatter(
    plot_idxs[~sorted_labels_selected_mask],
    suppixs_mean_ascores[~sorted_labels_selected_mask],
    marker=".",
    color="tab:red",
    label="not selected",
    s=2,
    alpha=.3,
)
_ = ax.axvline(
    np.max(np.where(sorted_labels_selected_mask)[0]),
    color="black", linestyle="--", label="best selection",
)

_ = ax.axhline(
    _get_aupimo_thresh_lower_bound(),
    color="tab:blue", linestyle="--", label="aupimo thresh",

)

_ = ax.set_ylabel("Anomaly score")
_ = ax.legend(loc="upper right")

ax = axrow[1]
_ = ax.scatter(
    plot_idxs[sorted_labels_selected_mask],
    suppixs_mean_ascores[sorted_labels_selected_mask],
    marker=".",
    color="tab:blue",
    label="selected",
    alpha=.4,
)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
_ = ax.scatter(
    plot_idxs[~sorted_labels_selected_mask],
    suppixs_mean_ascores[~sorted_labels_selected_mask],
    marker=".",
    color="tab:red",
    label="not selected",
    alpha=.1,
)
_ = ax.axhline(
    _get_aupimo_thresh_lower_bound(),
    color="tab:blue", linestyle="--", label="aupimo thresh",

)
_ = ax.set_xlim(xlim)
_ = ax.set_ylim(ylim)


# %%
suppixs_medasmap = np.copy(asmap.cpu().numpy())
for label in suppixs_labels:
    suppixs_medasmap[suppixs == label] = np.median(asmap[suppixs == label].cpu().numpy())

# %%
fig, ax = plt.subplots(1, 1, figsize=np.array((6, 6))*2)
_ = ax.imshow(suppixs_medasmap, cmap="jet")
_ = ax.contour(mask, levels=[0.5], colors="black", linewidths=2.5, linestyles="--")
_ = ax.contour(best_segmentation, levels=[0.5], colors="gray", linewidths=1.5)

# %%
