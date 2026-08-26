import os
import csv
import sys
from paths import ROOT_HPC, PYTHON_ENV, PATH_DOCUMENTS_FOLDER, PROJECT_PATH, STAGE_PATH, CONFIG_PATH
from utils import generate_config_id
sys.path.insert(0, PROJECT_PATH) 
#print(sys.path)
from manage_files.read_save_files import save_in_file, make_dir

def read_config_csv(config_path):
    with open(config_path, mode='r') as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames
        configs = [row for row in reader]
    return header, configs

def fill_config_with_default(config, idx):
    default_config = {
        "gen":              True,
        "rec":              True,
        "prof":             False,
        "gt_name":          None,
        "fourier":          False,
        "config_id":        generate_config_id(idx),
        "pth_views":        None,
        "sig_z":            5,
        "nb_views":         250,
        "snr":              1,
        "save_fold":        None,
        "nb_epochs":        3000,
        "x":                100,
        "timeout":          "1:00:00",
        "nb_hps":           50,
        "nb_kept":          8,
        "nb_children":      8,
        "nb_dir":           72,
        "nb_in_plane":      12,
        "scale_factors":    None
    }
    for key in default_config:
        if key not in config or config[key] == "":
            config[key] = default_config[key]
    return config

def build_cli_args(config):
    return (f" --gen                {config['gen']}"
            f" --rec                {config['rec']}"
            f" --prof               {config['prof']}"
            f" --gt_name            {config['gt_name']}"
            f" --fourier            {config['fourier']}"
            f" --config_id          {config['config_id']}"
            f" --pth_views          {config['pth_views']}"
            f" --sig_z              {config['sig_z']}"
            f" --nb_views           {config['nb_views']}"
            f" --snr                {config['snr']}"
            f" --save_fold          {config['save_fold']}"
            f" --nb_epochs          {config['nb_epochs']}"
            f" --x                  {config['x']}"
            f" --nb_hps             {config['nb_hps']}"
            f" --nb_kept            {config['nb_kept']}"
            f" --nb_children        {config['nb_children']}"
            f" --nb_dir             {config['nb_dir']}"
            f" --nb_in_plane        {config['nb_in_plane']}"
            f" --scale_factors      {config['scale_factors']}")

def write_slurm_file(save_fold, config):
    text = "#! /bin/bash \n"
    text += "#SBATCH -p publicgpu \n"
    text += "#SBATCH -A miv \n"
    text += "#SBATCH -N 1 \n"
    text += "#SBATCH -n 1 \n"
    text += "#SBATCH -c 1 \n"
    text += f"#SBATCH -t {config['timeout']} \n" 
    text += "#SBATCH --gres=gpu:1 \n"
    text += "#SBATCH --mem=16G \n"
    text += "#SBATCH --constraint='gpup100|gpurtx5000|gpua100|gpul40s' \n"
    text += f"#SBATCH -o {save_fold}/slurm.out \n"
    text += "\n"
    text += f"source {ROOT_HPC}/anaconda3/etc/profile.d/conda.sh \n"
    text += "conda activate single_particle_reconstruction \n"
    #text += f"CUDA_LAUNCH_BLOCKING=1 {PYTHON_ENV} {STAGE_PATH}/main.py " + build_cli_args(config)
    text += f"{PYTHON_ENV} {STAGE_PATH}/main.py " + build_cli_args(config)
    save_pth = f"{save_fold}/job.slurm"
    save_in_file(text, save_pth)

def run_slurm(save_fold):
    os.system(f"sbatch {save_fold}/job.slurm")

def run_python(config):
    os.system(f"{PYTHON_ENV} {STAGE_PATH}/main.py " + build_cli_args(config))

if __name__ == "__main__":
    header, configs = read_config_csv(CONFIG_PATH)
    for idx, config in enumerate(configs):  
        config = fill_config_with_default(config, idx)  
        save_fold = f"{PATH_DOCUMENTS_FOLDER}/jobs/{config['config_id']}"
        make_dir(save_fold)
        if config["save_fold"] is None or config["save_fold"] == "":
            config["save_fold"] = f"{save_fold}/reconstruction"
        if config["pth_views"] is None or config["pth_views"] == "":
            config["pth_views"] = f"{save_fold}/views"
        write_slurm_file(save_fold, config)
        
        #run_python(config)
        run_slurm(save_fold)