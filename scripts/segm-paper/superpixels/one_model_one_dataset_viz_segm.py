
fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True, constrained_layout=True)
_ = ax.set_xticks([])
_ = ax.set_yticks([])
plot_best_supperpixel_segmentation(ax, img, mask.numpy(), superpixel_best_segmentation, superpixel_best_iou)

def plot_best_supperpixel_segmentation(
    ax: plt.Axes,
    image: np.ndarray,
    mask: np.ndarray,
    best_segmentation: np.ndarray,
    best_iou: float,
) -> None:
    """TODO(jpcbertoldo): write docstring of `plot_best_supperpixel_segmentation`."""
    _ = ax.imshow(image)
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=2.5,
        linestyles="--",
    )
    cs_gt = ax.contour(
        best_segmentation == 1,
        levels=[0.5],
        colors="magenta",
        linewidths=1,
        linestyles="-",
    )
    _ = ax.annotate(
        f"best iou={best_iou:.0%}",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(10, -10),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=16,
        bbox=dict(  # noqa: C408
            facecolor="white",
            alpha=1,
            edgecolor="black",
            boxstyle="round,pad=0.2",
        ),
    )
