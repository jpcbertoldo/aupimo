
MODELS_LABELS_LONG = {
"padim_r18": "PaDiM ResNet18",
"padim_wr50": "PaDiM WideResNet50",
"patchcore_wr50": "PatchCore WideResNet50",
"patchcore_wr101": "PatchCore WideResNet101",
"fastflow_wr50": "FastFlow WideResNet50",
"fastflow_cait_m48_448": "FastFlow CAIT M48",
"efficientad_wr101_s_ext": "EfficientAD WideResNet101 Small",
"efficientad_wr101_m_ext": "EfficientAD WideResNet101 Medium",
"simplenet_wr50_ext": "SimpleNet WideResNet50",
"pyramidflow_fnf_ext": "PyramidFlow Fully Normalized Flow",
"pyramidflow_r18_ext": "PyramidFlow ResNet18",
"uflow_ext": "U-Flow",
"rd++_wr50_ext": "Reverse Distillation++ WideResNet50",
}

MODELS_LABELS = {
"padim_r18": "PaDiM R18",
"padim_wr50": "PaDiM WR50",
"patchcore_wr50": "PatchCore WR50",
"patchcore_wr101": "PatchCore WR101",
"fastflow_wr50": "FastFlow WR50",
"fastflow_cait_m48_448": "FastFlow CAIT",
"efficientad_wr101_s_ext": "EfficientAD S",
"efficientad_wr101_m_ext": "EfficientAD M",
"simplenet_wr50_ext": "SimpleNet WR50",
"pyramidflow_fnf_ext": "PyramidFlow FNF",
"pyramidflow_r18_ext": "PyramidFlow R18",
"uflow_ext": "U-Flow",
"rd++_wr50_ext": "RevDist++ WR50",
}

MODELS_LABELS_SHORT = {
"padim_r18": "PDM R18",
"padim_wr50": "PDM WR50",
"patchcore_wr50": "PC WR50",
"patchcore_wr101": "PC WR101",
"fastflow_wr50": "FF WR50",
"fastflow_cait_m48_448": "FF CAIT",
"efficientad_wr101_s_ext": "Eff S",
"efficientad_wr101_m_ext": "Eff M",
"simplenet_wr50_ext": "SN WR50",
"pyramidflow_fnf_ext": "PF FNF",
"pyramidflow_r18_ext": "PF R18",
"uflow_ext": "UF",
"rd++_wr50_ext": "RD++ WR50",
}

MODELS_LABELS_MAINTEXT = {
"padim_r18": "PaDiM",
"padim_wr50": "PaDiM",
"patchcore_wr50": "PatchCore",
"patchcore_wr101": "PatchCore",
"fastflow_wr50": "FastFlow",
"fastflow_cait_m48_448": "FastFlow",
"efficientad_wr101_s_ext": "EfficientAD",
"efficientad_wr101_m_ext": "EfficientAD",
"simplenet_wr50_ext": "SimpleNet",
"pyramidflow_fnf_ext": "PyramidFlow",
"pyramidflow_r18_ext": "PyramidFlow",
"uflow_ext": "U-Flow",
"rd++_wr50_ext": "RevDist++",
}

DATASETS_LABELS = {
    'visa': "VisA",
    'mvtec': "MVTec AD",
}

DATALABELS_SHORT = {
    'visa': "V",
    'mvtec': "M",
}

CATEGORIES_LABELS = {
    'bottle': "Bottle",
    'cable': "Cable",
    'candle': "Candle",
    'capsule': "Capsule",
    'capsules': "Capsules",
    'carpet': "Carpet",
    'cashew': "Cashew",
    'chewinggum': "Chewing Gum",
    "fryum": "Fryum",
    'grid': "Grid",
    'hazelnut': "Hazelnut",
    'leather': "Leather",
    'macaroni1': "Macaroni 1",
    'macaroni2': "Macaroni 2",
    'metal_nut': "Metal Nut",
    'pcb1': "PCB 1",
    'pcb2': "PCB 2",
    'pcb3': "PCB 3",
    'pcb4': "PCB 4",
    'pipe_fryum': "Pipe Fryum",
    'pill': "Pill",
    'screw': "Screw",
    'tile': "Tile",
    'toothbrush': "Toothbrush",
    'transistor': "Transistor",
    'wood': "Wood",
    'zipper': "Zipper",
}

CATEGORIES_LABELS_SHORT = {
    'bottle': "Bot",
    'cable': "Cab",
    'candle': "Cand",
    'capsule': "Cap",
    'capsules': "Caps",
    'carpet': "Carp",
    'cashew': "Cash",
    'chewinggum': "ChewG",
    "fryum": "Fry",
    'grid': "Grid",
    'hazelnut': "Haz",
    'leather': "Leat",
    'macaroni1': "Mac1",
    'macaroni2': "Mac2",
    'metal_nut': "MetNu",
    'pcb1': "PCB1",
    'pcb2': "PCB2",
    'pcb3': "PCB3",
    'pcb4': "PCB4",
    'pipe_fryum': "PiFry",
    'pill': "Pill",
    'screw': "Scr",
    'tile': "Tile",
    'toothbrush': "Toot",
    'transistor': "Trans",
    'wood': "Wood",
    'zipper': "Zip",
}

DS_CAT_COMBINATIONS = [
    ("mvtec", "bottle"),
    ("mvtec", "cable"),
    ("mvtec", "capsule"),
    ("mvtec", "carpet"),
    ("mvtec", "grid"),
    ("mvtec", "hazelnut"),
    ("mvtec", "leather"),
    ("mvtec", "metal_nut"),
    ("mvtec", "pill"),
    ("mvtec", "screw"),
    ("mvtec", "tile"),
    ("mvtec", "toothbrush"),
    ("mvtec", "transistor"),
    ("mvtec", "wood"),
    ("mvtec", "zipper"),
    ("visa", "candle"),
    ("visa", "capsules"),
    ("visa", "cashew"),
    ("visa", "chewinggum"),
    ("visa", "fryum"),
    ("visa", "macaroni1"),
    ("visa", "macaroni2"),
    ("visa", "pcb1"),
    ("visa", "pcb2"),
    ("visa", "pcb3"),
    ("visa", "pcb4"),
    ("visa", "pipe_fryum"),
] 

DC_LABELS = {
    f"{ds}/{cat}": f"{DATASETS_LABELS[ds]} / {CATEGORIES_LABELS[cat]}"
    for ds, cat in DS_CAT_COMBINATIONS
}

DC_LABELS_SHORT = {
    f"{ds}/{cat}": f"{DATALABELS_SHORT[ds]} / {CATEGORIES_LABELS_SHORT[cat]}"
    for ds, cat in DS_CAT_COMBINATIONS
}

METRICS_LABELS = {
    'auroc': "AUROC",
    'aupr': "AUPR",
    'aupro': 'AUPRO',
    "aupro_05": "AUPRO 5%",
    "aupro5": "AUPRO 5%",
    "avg_aupimo": "Avg. AUPIMO",
    "std_aupimo": "Std. AUPIMO",
    "p33_aupimo": "P33 AUPIMO",
    "med_aupimo": "Median AUPIMO",
    "avgrank_aupimo": "Avg. Rank",
}

METRICS_COLORS = {
    "auroc": "tab:blue",
    "aupro": "tab:red",
    "aupro5": "tab:purple",
    "aupimo": "tab:green",
    "avg_aupimo": "tab:green",
}

MAINTEXT_MODELS = [
    "padim_r18",
    # "padim_wr50",
    # "patchcore_wr50",
    "patchcore_wr101",
    # "fastflow_wr50",
    "fastflow_cait_m48_448",
    # "efficientad_wr101_s_ext",
    "efficientad_wr101_m_ext",
    "simplenet_wr50_ext",
    "pyramidflow_fnf_ext",
    # "pyramidflow_r18_ext",
    "uflow_ext",
    "rd++_wr50_ext",
] 


# RCPARAMS = {
#     "font.family": "sans-serif",
#     "axes.titlesize": "xx-large",
#     "axes.labelsize": 'large',
#     "xtick.labelsize": 'large',
#     "ytick.labelsize": 'large',
# }