"""TODO to be reviewd entirely, maybe break it into two scripts, one for the curves only"""
# %%
# AUX DF (IMGPATHS, MASKSPATHS)


IMGPATHS = data.reset_index()[["dataset", "category"]].drop_duplicates().reset_index(drop=True)
IMGPATHS["dir"] = IMGPATHS.apply(
    lambda row: common.get_dataset_category_testimg_dir(row["dataset"], row["category"]),
    axis=1,
)
IMGPATHS["imgpath"] = IMGPATHS["dir"].apply(Path).apply(
    lambda d: sorted(d.glob("**/*.png")) + sorted(d.glob("**/*.JPG"))
).apply(lambda ps: [(idx, str(p)) for idx, p in enumerate(ps)])
IMGPATHS = IMGPATHS.drop(columns="dir").explode("imgpath").set_index(["dataset", "category"])
IMGPATHS = IMGPATHS.apply(
    lambda row: {"imgidx": row.imgpath[0], "imgpath": row.imgpath[1]}, axis=1, result_type='expand'
).reset_index().set_index(IMGPATHS.index.names + ["imgidx"])["imgpath"]

print("aux df IMGPATHS")
IMGPATHS.head(2)

MASKSPATHS = IMGPATHS.copy().apply(
    lambda p: (
        DATASETSDIR / relpath
        if (relpath := common.imgpath_2_maskpath(p.split("/datasets/")[1])) is not None
        else None
    )
).rename("maskpath")

print("aux df MASKSPATHS")
MASKSPATHS.head(2)

# %%
# DC ARGS / DC BEST MODEL ARGS

# `dc` stands for dataset/category
dc_args = data.reset_index()[cols := ["dataset", "category"]].drop_duplicates().sort_values(cols).reset_index(drop=True)
dc_args.columns.name = None

records = []

# `dc` stands for dataset/category
for dcidx, dcrow in dc_args.iterrows():

    print(f"{dcidx=} {dcrow.dataset=} {dcrow.category=}")
    data = data.loc[dcrow.dataset, dcrow.category]

    # RANK DIAGRAM DATA
    nonparametric = compare_models_pairwise_wilcoxon(data["aupimo"].to_dict(), higher_is_better=True, alternative="greater")

    rank_avgs = nonparametric.reset_index("model1").index.values.tolist()
    rank_models_ordered = nonparametric.reset_index("Average Rank").index.values.tolist()  # models order!!!!
    best_model = rank_models_ordered[0]

    records.append({
        **dcrow.to_dict(),
        "best_model": best_model,
    })

dc_bestmodel_args = pd.DataFrame.from_records(records)
dc_bestmodel_args

# %%
# HEATMAPS and PIMO CURVES

set_ipython_autoreload_2()

from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import matplotlib as mpl  # noqa: E402  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from anomalib.post_processing.post_process import get_contour_from_mask  # noqa: E402
from anomalib.utils.metrics.perimg import perimg_boxplot_stats  # noqa: E402
from anomalib.utils.metrics.perimg.pimo import AUPImOResult, PImOResult  # noqa: E402
from anomalib.utils.metrics.perimg.plot import _plot_perimg_curves  # noqa: E402
from common import DATADIR  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from progressbar import progressbar  # noqa: E402


def get_binary_cmap(colorpositive="red"):
    return mpl.colors.ListedColormap(['#00000000', colorpositive])

STATS_LABELS = dict(
    mean="Mean", whislo="Lower Whisker", q1="Q1", med="Median", q3="Q3", whishi="Upper Whisker",
)

def get_image_label(stat_name, value, imgidx):
    return f"{STATS_LABELS[stat_name]}: {value:.0%} ({imgidx:03})"

INSTANCES_COLORMAP = mpl.colormaps["tab10"]

def plot_boxplot_logpimo_curves(shared_fpr, tprs, image_classes, stats, ax):
    imgidxs = [dic["imgidx"] for dic in stats.values()]

    _plot_perimg_curves(
        ax, shared_fpr, tprs[imgidxs],
        *[
            dict(
                # label=f"{STATS_LABELS[stat]}: {dic['value']:.0%} ({dic['imgidx']:03})",
                label=get_image_label(stat, dic["value"], dic["imgidx"]),
                color=INSTANCES_COLORMAP(curvidx),
                lw=3,
            )
            for curvidx, (stat, dic) in enumerate(stats.items())
        ]
    )
    # x-axis
    _ = ax.set_xlabel("Avg. Nor. Img. FPR (log)")
    _ = ax.set_xscale("log")
    _ = ax.set_xlim(1e-5, 1)
    _ = ax.set_xticks(np.logspace(-5, 0, 6, base=10))
    _ = ax.set_xticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
    _ = ax.minorticks_off()
    _ = ax.axvspan(1e-5, 1e-4, color="tab:blue", alpha=0.3, zorder=-5, label="AUC range")
    # y-axis
    _ = ax.set_ylabel("Per-Image TPR")
    _ = ax.set_ylim(0, 1.02)
    _ = ax.set_yticks(np.linspace(0, 1, 11))
    _ = ax.set_yticklabels(["0", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])
    _ = ax.grid(True, axis="y", zorder=-10)
    #
    # _ = ax.set_title("PIMO Curves (AUC boxplot statistics)")
    ax.legend(title="[Statistic]: [AUPIMO] ([Image Index])", loc="lower right")


# `dc` stands for dataset/category
for dcidx, dcrow in dc_bestmodel_args.iterrows():

    dataset, category = dcrow[["dataset", "category"]]
    best_model = dcrow["best_model"]

    # DATASET DF
    data = data.loc[dataset, category]

    # MODEL DF
    modeldf = data.loc[dataset, category, best_model]

    aupimos = torch.tensor(modeldf["aupimo"])
    asmaps = torch.tensor(torch.load(modeldf["asmaps_path"]))  # some where save as numpy array

    curves = PImOResult.load(modeldf["aupimo_curves_fpath"])
    aucurves = AUPImOResult.load(modeldf["aupimo_aucs_fpath"])

    imgclass = curves.image_classes

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if best_model == "rd++_wr50_ext":
        argsort = np.argsort((Path(modeldf.dir) / "key.txt").read_text().splitlines())
        asmaps = asmaps[argsort]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # lower/upper fpr bound is upper/lower threshold bound
    upper_threshold = curves.threshold_at(aucurves.lbound).item()
    lower_threshold = curves.threshold_at(aucurves.ubound).item()

    boxplot_stats_dicts = perimg_boxplot_stats(aupimos, imgclass, only_class=1, repeated_policy="avoid")
    stats = {
        stat_dict["statistic"]: {
            "value": stat_dict["nearest"],
            "imgidx": stat_dict["imgidx"],
        }
        for stat_dict in boxplot_stats_dicts
    }
    stats = OrderedDict([
        (k, stats[k])
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # images order in the paper
        for k in ["mean", "whislo", "q1", "med", "q3", "whishi"]
    ])

    # =======================================================================
    # HEATMAPS

    (heatmaps_dc_savedir := HEATMAPS_SAVEDIR / f"{dcidx:03}").mkdir(exist_ok=True, parents=True)

    for vizidx, (stat_name, stat_dict) in enumerate(stats.items()):

            imgidx = stat_dict["imgidx"]
            imgpil = Image.open(imgpath := IMGPATHS.loc[dataset, category, imgidx]).convert("RGB")
            maskpil = Image.open(maskpath := MASKSPATHS.loc[dataset, category, imgidx]).convert("L")
            assert (resolution := imgpil.size[::-1]) == maskpil.size[::-1]
            # [::-1] makes it (height, width)

            asmap = asmaps[imgidx]
            asmap_fullsize = torch.nn.functional.interpolate(
                asmap.unsqueeze(0).unsqueeze(0),
                size=resolution,
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
            img = np.asarray(imgpil)
            gt = np.array(maskpil).astype(bool)

            fig, ax = plt.subplots(figsize=(sz := 6, sz), dpi=150, layout="constrained")
            _ = ax.imshow(img)
            asmap_viz = asmap_fullsize.copy()
            asmap_viz[asmap_fullsize < lower_threshold] = np.nan
            asmap_viz[asmap_fullsize > upper_threshold] = np.nan
            cmap_pimo_range = mpl.colors.ListedColormap(mpl.colormaps["Blues"](np.linspace(.6, 1, 1024)))
            cmap_pimo_range.set_bad("white", alpha=0)
            _ = ax.imshow(
                asmap_viz,
                cmap=cmap_pimo_range, alpha=0.4, vmin=lower_threshold, vmax=upper_threshold,
            )
            asmap_viz = asmap_fullsize.copy()
            asmap_viz[asmap_fullsize < upper_threshold] = np.nan
            cmap_anomaly_range = mpl.colors.ListedColormap(mpl.colormaps["Reds"](np.linspace(.4, 1, 1024)))
            cmap_anomaly_range.set_bad("white", alpha=0)
            ascores_beyond_ubound = asmap > upper_threshold
            ascores_vmax = (
                torch.quantile(asmap[ascores_beyond_ubound], 0.99)
                if ascores_beyond_ubound.any()
                else upper_threshold + 1e-5
            )
            _ = ax.imshow(
                asmap_viz,
                cmap=cmap_anomaly_range, alpha=.4, vmin=upper_threshold, vmax=ascores_vmax,
            )
            _ = ax.contour(
                get_contour_from_mask(gt, square_size=3, type="outter").astype(float),
                cmap=get_binary_cmap("white"),
            )
            _ = ax.annotate(
                get_image_label(stat_name, stat_dict["value"], stat_dict["imgidx"]),
                xy=((eps := 40), eps), xycoords='data',
                xytext=(0, 0), textcoords='offset fontsize',
                va="top", ha="left",
                fontsize=22,
                bbox=dict(facecolor='white', edgecolor='black')
            )
            _ = ax.axis("off")
            border = np.ones_like(gt).astype(bool)
            border = cv2.dilate(border.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
            _ = ax.contour(
                # get_contour_from_mask(, square_size=5, type="inner").astype(float),
                border.astype(float),
                cmap=get_binary_cmap("black"),
            )
            # add a square to frame the image
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            _ = ax.add_patch(
                mpl.patches.Rectangle((0, 0), *(np.array(gt.shape)[::-1]),
                linewidth=15, edgecolor=INSTANCES_COLORMAP(vizidx), facecolor='none')
            )
            _ = ax.set_xlim(xlim)
            _ = ax.set_ylim(ylim)
            fig
            fig.savefig(heatmaps_dc_savedir / f"{vizidx:03}.jpg", bbox_inches="tight")
            raise Exception("stop")
            break
    break
    continue

    # =======================================================================
    # PIMO CURVES
    with mpl.rc_context(rc=RCPARAMS):
        fig_curves, ax = plt.subplots(figsize=(7, 4), dpi=100, layout="constrained")
        plot_boxplot_logpimo_curves(curves.shared_fpr, curves.tprs, curves.image_classes, stats, ax)
        fig_curves
        fig_curves.savefig(PERDATASET_SAVEDIR / f"perdataset_{dcidx:03}_curves.pdf", bbox_inches="tight")

    break

print('done')

