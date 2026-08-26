import argparse
from datetime import datetime
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--prof", type=str2bool, default=False)
    parser.add_argument("--gt_name", type=str, default=None)
    parser.add_argument("--fourier", type=str2bool, default=False)
    parser.add_argument("--config_id", type=str, default=None)  
    parser.add_argument("--pth_views", type=str, default=None)
    parser.add_argument("--sig_z", type=float, default=5)
    parser.add_argument("--nb_views", type=int, default=250)
    parser.add_argument("--snr", type=float, default=1)
    parser.add_argument("--save_fold", type=str, default=None)
    parser.add_argument("--nb_epochs", type=int, default=3000)
    parser.add_argument("--x", type=int, default=100)
    parser.add_argument("--nb_hps", type=int, default=50)
    parser.add_argument("--nb_kept", type=int, default=8)
    parser.add_argument("--nb_children", type=int, default=8)
    parser.add_argument("--nb_dir", type=int, default=72)
    parser.add_argument("--nb_in_plane", type=int, default=12)
    parser.add_argument("--scale_factors", type=scales_param, default=None)
    return parser.parse_args()

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected: {value}")

def scales_param(value):
    if value == "None":
        return None
    scale_factors = value[1:-1].split(",")
    parsed = []
    for item in scale_factors:
        item = item.strip()
        if item == "" or item == "None":
            parsed.append(None)
        else:
            parsed.append(float(item))
    return parsed
    
def p2p_param(value):
    if value == "None":
        return None
    regex = r"(\d+)"
    matches = re.findall(regex, value)
    if len(matches) != 2:
        raise argparse.ArgumentTypeError(f"p2p parameter must be either None or in the form '[n, m]' where n and m are integers")
    return [int(matches[0]), int(matches[1])]

def generate_config_id(idx):
    now = datetime.now()
    config_id = now.strftime("%Y%m%d_%H%M%S") + f"_{idx}"
    return config_id