import json,os

ARGS_PATH = "./args.json"
DATASET_CFG_PATH = "./datasets/dataset_cfg.json"

with open(DATASET_CFG_PATH) as f:
    _cfg = json.load(f)
    DATASET_PARAMS = _cfg["default_dataset_params"]  
    DATASET_PATHS = _cfg["dataset_paths"]
    DATASET_LIST = _cfg["datasets"]

DOWNSAMPLE_RATIO = 8

def load_args(args_path):
    with open(args_path) as f:
        args = json.load(f)
    return args