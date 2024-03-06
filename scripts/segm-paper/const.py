MODEL_2_LABEL = {
    # "efficientad_wr101_m_ext": "EfficientAD-M",
    # "efficientad_wr101_s_ext": "EfficientAD-S",
    "efficientad_wr101_s_ext": "EfficientAD",
    "patchcore_wr50": "PatchCore",
    # "patchcore_wr50": "PatchCore WR50",
    # "patchcore_wr101": "PatchCore WR101",
    # "rd++_wr50_ext": "RevDist++ WR50",
    "rd++_wr50_ext": "RevDist++",
    "uflow_ext": "U-Flow",
}


COLLECTION_2_LABEL = {
    "mvtec": "MVTec-AD",
    "visa": "VisA",
}

DATASET_2_LABEL = {
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

METHOD_2_LABEL = {
    'global_thresh': "Oracle Global Thresh.",
    'local_thresh': "Oracle Per-Image Thresh.",
    'local_thresh_min_val': "Oracle Per-Image Thresh (with val.)",
    'heuristic_best': "Heuristic (best)",
}
