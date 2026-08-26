from manage_files.read_save_files import save_in_file, make_dir, save_pickle
from reconstruction_with_dl.test_params.default_params import params_data_gen, params_learn_setup
from manage_files.paths import PATH_PROJECT_FOLDER
import os


root_hpc = "/home2020/home/miv/teloy"


def write_slurm_file(save_fold, save_name, pth_param_learn_setup, pth_param_data_gen):
    text = "#! /bin/bash \n"
    text += "#SBATCH -p publicgpu -A miv \n"
    text += "#SBATCH -N 1 \n"
    text += "#SBATCH -n 1 \n"
    text += "#SBATCH -c 1 \n"
    text += "#SBATCH --gres=gpu:1 \n"
    text += "#SBATCH --mem=16G \n"
    text += "#SBATCH --constraint='gpua100|gpua40|gpurtx6000|gpurtx5000|gpuv100|gpup100|gpuk80' \n"
    text += f"#SBATCH -o {save_fold}/jobs/slurm.out \n"
    text += "\n"
    text += f"source {root_hpc}/anaconda3/etc/profile.d/conda.sh \n"
    text += "conda activate pytorch3d \n"
    text += "conda develop . \n"
    text += (f"python3 {root_hpc}/code/reconstruction_with_dl/FluoFire.py "
             f"--pth1 {pth_param_learn_setup} --pth2 {pth_param_data_gen}")
    save_pth = f'{save_fold}/{save_name}'
    save_in_file(text, save_pth)


def test_slurm(param_to_test, values_to_test, nb_test_by_point, params_learn_setup,
               params_data_gen, save_fold_slurm_files, save_fold_results, nb_channels=2, plus_to_name=''):
    params_learn_setup["x"] = params_learn_setup["nb_epochs"] -1
    params_learn_setup["nb_channels"] = nb_channels
    for val in values_to_test:
        save_fold_val = f'{save_fold_results}/{param_to_test}_{val}_{plus_to_name}'
        make_dir(save_fold_val)
        params_data_gen[param_to_test] = val
        save_pickle(params_data_gen, save_fold_val, "params_data_gen")
        for t in range(nb_test_by_point):
            save_fold_val_test = f'{save_fold_val}/test_{t}'
            make_dir(save_fold_val_test)
            params_learn_setup["save_fold"] = save_fold_val_test
            params_learn_setup["device"] = 0
            save_pickle(params_learn_setup, save_fold_val_test, "params_learn_setup")
            inter_save_fold_slurm = f'{save_fold_slurm_files}/{param_to_test}_{val}_{plus_to_name}/test_{t}'
            make_dir(f'{inter_save_fold_slurm}/jobs')
            write_slurm_file(inter_save_fold_slurm, 'fluo_fire.slurm',
                             f'{save_fold_val_test}/params_learn_setup',
                             f'{save_fold_val}/params_data_gen')
            os.system(f'sbatch {inter_save_fold_slurm}/fluo_fire.slurm')


pth_results = f'{root_hpc}/results_tests/test_anis_2/recons_results'
pth_slurm_files = f'{root_hpc}/results_tests/test_anis_2/slurm_files'
params_data_gen["nb_views"] = 250
params_learn_setup["nb_channels"] = 2
params_learn_setup["nb_epochs"] = 3001
params_learn_setup["x"] = 3000
params_learn_setup["nb_epochs_each_phases_ACE_Het"] = None

for sig_z in [1,3,5,10,15]:
    params_data_gen["sig_z"] = sig_z
    test_slurm('nb_views', [5,10,20,50], 5, params_learn_setup, params_data_gen, pth_slurm_files,
           pth_results, plus_to_name=f'_sig_z_{sig_z}')



