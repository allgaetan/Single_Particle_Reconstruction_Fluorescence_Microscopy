import os

from learning_algorithms.gradient_descent_importance_sampling import gd_importance_sampling_3d
from learning_algorithms.gradient_descent_known_angles import gradient_descent_known_rot
from volume_representation.pixel_representation import Fourier_pixel_representation
from classes_with_parameters import ParametersMainAlg, ParametersDataGeneration
from manage_files.read_save_files import read_image, read_images_in_folder, read_csv, return_het_with_name
from manage_files.paths import PATH_PROJECT_FOLDER
from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly
import numpy as np
from manage_files.read_save_files import make_dir
from common_image_processing_methods.others import crop_center

nb_dim = 3
size = 45
N_iter = 30
fold_in = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/splitted_in_subsets'
fold_out = f'{PATH_PROJECT_FOLDER}/results/splitted_in_subsets_adjusted_fixed_angles'
fold_angles = f'{PATH_PROJECT_FOLDER}/results/splitted_in_subsets'
params_data_gen = ParametersDataGeneration(sig_xy=1, sig_z=3, size=size)
psf = params_data_gen.get_psf()
#psf = crop_center(psf, (size, size, size))
params_learning_alg = ParametersMainAlg(eps=-10, N_iter_max=N_iter, lr=0.1)


for sub_fold in os.listdir(fold_in):
    if sub_fold == 'set3':
        sub_fold_ = f'{fold_in}/{sub_fold}'
        views, file_names = read_images_in_folder(sub_fold_, sort_fn=return_het_with_name)
        true_trans_vecs = np.zeros((len(views), 3))
        print('sub fold', sub_fold)
        print('file names', file_names)
        make_dir(sub_fold_)
        volume_representation = Fourier_pixel_representation(nb_dim, size, psf)
        uniform_sphere_discretization = discretize_sphere_uniformly(params_learning_alg.M_axes, params_learning_alg.M_rot)
        imp_distrs_rot = np.ones((len(views), params_learning_alg.M_rot)) / params_learning_alg.M_rot
        imp_distrs_axes = np.ones((len(views), params_learning_alg.M_axes)) / params_learning_alg.M_axes
        gd_importance_sampling_3d(volume_representation, uniform_sphere_discretization, true_trans_vecs, views,
                                  imp_distrs_axes,
                                  imp_distrs_rot, 1, params_learning_alg.prop_min, params_learning_alg, False, f'{fold_out}/{sub_fold}',
                                  use_gpu=True)
    """
    rot_vecs = read_csv(f'{fold_angles}/{sub_fold}/rot_vecs_adjusted.csv')
    volume_representation = Fourier_pixel_representation(nb_dim, size, psf)
    gradient_descent_known_rot(volume_representation, rot_vecs, true_trans_vecs, views, params_learning_alg, False,
                               f'{fold_out}/{sub_fold}_fixed_angles')
    """