import ast

from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import (make_grid, nd_gaussian,
                                                                                       gaussian_mixture_isotrop_identical_gaussians)
from manage_files.read_save_files import read_csv, load_pickle
import numpy as np
from manage_files.read_save_files import save, make_dir, save_4d_for_chimera
import tifffile
from manage_files.paths import PATH_PROJECT_FOLDER
from data_generation.generate_data import generate_one_view, generate_rot_vec_trans_vec
from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly



def seprates_npc(NPCs):
    """trakes NPC arrays in the form of 2D tables, with last row representing the index of NPC and returns a 3D array
    with NPC separated"""
    NPCs_separated = []
    nb_NPCs = int(NPCs[-1][-1] + 1)
    for i in range(nb_NPCs):
        indices = np.where(NPCs[:, -1] == i)
        NPC_i = NPCs[indices, :3]
        NPCs_separated.append(NPC_i)
    NPCs_separated = np.array(NPCs_separated).squeeze()
    return NPCs_separated


def transorm_npc_to_image(npc, norm_factor, size = 50, sig=0.1):
    npc/=norm_factor
    grid = make_grid(size, 3)
    NPC_image = gaussian_mixture_isotrop_identical_gaussians(grid, [1] * len(npc), npc, sig, 3, 3)
    return NPC_image


def get_norm_factor(NPCs_separated):
    ma = max(np.max(NPCs_separated), -np.min(NPCs_separated))
    norm = 1.5 * ma
    return norm


def generates_views(NPCs_image, save_fold, params_data_gen):
    make_dir(save_fold)
    nb_views = len(NPCs_image)
    unif_sphere_discr = discretize_sphere_uniformly(10000, 360)
    for v in range(nb_views):
        rot_vec, t = generate_rot_vec_trans_vec(params_data_gen, unif_sphere_discr, False)
        view, _, _, _ = generate_one_view(NPCs_image[v], rot_vec, t, 3, params_data_gen)
        save(f'{save_fold}/view_{v}_{rot_vec[0]}_{rot_vec[1]}_{rot_vec[2]}_.tif', view)


if __name__ == '__main__':
    from classes_with_parameters import ParametersDataGeneration
    from common_image_processing_methods.others import *

    nb_frames = 300
    size = 50
    anis = 5
    alpha = 1
    sigma_trans_ker = 0
    params_data_gen = ParametersDataGeneration(sig_xy=alpha, sig_z=anis * alpha, nb_views=nb_frames, size=size,
                                               convention='ZXZ',
                                               no_psf=False, snr=10000, rotation_max=[180, 180, 180],
                                               sigma_trans_ker=sigma_trans_ker)
    NPCs_all_frames = load_pickle("NPC_coords_all_frames_other_params")
    NPCs_all_frames_separated = []

    for i in range(nb_frames):
        NPC_sep = seprates_npc(NPCs_all_frames[i])
        NPCs_all_frames_separated.append(NPC_sep)

    NPCs_all_frames_separated = np.array(NPCs_all_frames_separated)
    save_fold_root = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/NPC'
    for i in range(6):
        NPC_all_frames_i = NPCs_all_frames_separated[:, i, :, :]
        NPC_image_i = []
        norm_factor = get_norm_factor(NPC_all_frames_i)
        for frame in range(nb_frames):
            im = transorm_npc_to_image(NPC_all_frames_i[frame], norm_factor, sig=0.05)
            NPC_image_i.append(im)
        save_fold = f'{save_fold_root}/example_other_param_{i}_sig_{anis}'
        generates_views(NPC_image_i, f'{save_fold}/views/c1', params_data_gen)
        save_4d_for_chimera(NPC_image_i, f'{save_fold}/ground_truth.tiff')

    1/0



    NPCs = read_csv("NPC_coords.csv")
    NPCs_separated = seprates_npc(NPCs)
    all_NPC_image = []
    norm_factor = get_norm_factor(NPCs_separated)
    for i in range(len(NPCs_separated)):
        npc = NPCs_separated[i]
        NPC_image = transorm_npc_to_image(npc, norm_factor)
        all_NPC_image.append(NPC_image)

    save_4d_for_chimera(all_NPC_image)




