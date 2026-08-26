from volume_representation.gaussian_mixture_representation.GMM_representation import GMM_representation
from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly
from learning_algorithms.gradient_descent_importance_sampling import gd_importance_sampling_3d, gradient_descent_importance_sampling_known_axes,\
    gradient_descent_importance_sampling_known_rot
import numpy as np
import copy as cp
from learning_algorithms.gradient_descent_known_angles import gradient_descent_known_rot
from time import time
import matplotlib.pyplot as plt
from data_generation.generate_data import generate_and_save_data, read_views, generate_data
from metrics_and_visualisation.error_orientation_translation import plot_importance_distributions_2, plot_energies
from manage_files.read_save_files import *
from metrics_and_visualisation.error_orientation_translation import mean_error_true_shift
from manage_files.paths import PTH_LOCAL_RESULTS
from classes_with_parameters import *
from metrics_and_visualisation.metrics_to_compare_2_images import *
from manage_matplotlib.plot_graph import plot_experiment_graph
from metrics_and_visualisation.plot_conical_fsc import plot_conical_fsc
from metrics_and_visualisation.metrics_to_compare_2_images import ssim, fsc
from time import time


fold_gt = "../../../ground_truths"
for t in range(14,23):
    for gt in ['recepteurs_AMPA.tif', "HIV-1-Vaccine_prep.tif", "clathrine.tif", "emd_0680.tif"]:
        sig = 0.03
        nb_dim = 3
        size = 50
        params_learning_alg = ParametersMainAlg(lr=2*10 ** -4, N_iter_max=30, eps=-5, N_axes=25, N_rot=20)
        params_data_gen = ParametersDataGeneration(nb_views=20)
        cov_PSF = params_data_gen.get_cov_psf()
        params_gmm = ParametersGMM(nb_gaussians_init=250, nb_steps=1, sigma_init=sig, init_with_views=True,
                                   threshold_gaussians=0.01)
        folder_results = f'{PTH_LOCAL_RESULTS}/gmm_test/{gt}/test_unknown_angles_{t}'
        folder_views = f'{folder_results}/views'
        gt_im = generate_and_save_data(folder_views, fold_gt, gt, params_data_gen)
        make_dir(folder_results)
        params = locals()
        views, rot_vecs, trans_vecs, file_names = read_views(folder_views, params_data_gen.nb_dim)
        gmm_rep = GMM_representation(params_gmm.nb_gaussians_init, params_gmm.sigma_init, nb_dim, size, cov_PSF, params_gmm.threshold_gaussians)
        if params_gmm.init_with_views:
            gmm_rep.init_with_average_of_views(views)
        gmm_rep.register_and_save(folder_results, 'init_gmm_vol')
        uniform_sphere_discretization = discretize_sphere_uniformly(params_learning_alg.M_axes, params_learning_alg.M_rot)
        imp_distrs_axes = np.ones((len(views), params_learning_alg.M_axes)) / params_learning_alg.M_axes
        imp_distrs_rot = np.ones((len(views), params_learning_alg.M_rot)) / params_learning_alg.M_rot
        t0 = time()
        imp_distrs_rot_recorded, imp_distrs_axes_recorded, recorded_energies, recorded_shifts, unif_prop, gmm_rep, itr, _, _, _, _ = \
                        gd_importance_sampling_3d(gmm_rep, uniform_sphere_discretization,
                                              trans_vecs, views, imp_distrs_axes,
                                              imp_distrs_rot, 1, 0, params_learning_alg, True, folder_results)
        temps_calcul = time() - t0
        gmm_rep.save_gmm_parameters(folder_results)
        recons_voxel = gmm_rep.register_and_save(folder_results, 'recons', ground_truth_path=f'{fold_gt}/{gt}')
        mean_errors_angle, _, angles_found = \
                    plot_importance_distributions_2(folder_results, rot_vecs, imp_distrs_axes_recorded, imp_distrs_rot_recorded,
                                                    params_data_gen.nb_dim, None, uniform_sphere_discretization, None,
                                                    params_data_gen.convention,
                                                    partition_rot_graphs=[range(10), range(10,20)])
        plot_energies(recorded_energies, folder_results)
        write_array_csv(recorded_energies, f'{folder_results}/energies.csv')
        write_array_csv(mean_errors_angle, f'{folder_results}/error_angles.csv')
        print('gmm rep shape', recons_voxel.shape)
        print('gt im shape', gt_im.shape)
        plot_conical_fsc(recons_voxel, gt_im, f'{folder_results}/cfsc.png')
        ssim_val = ssim(recons_voxel, gt_im)
        fsc_val = fsc(recons_voxel, gt_im, cutoff=0.143)
        write_array_csv(np.array([[ssim_val]]), f'{folder_results}/ssim.csv')
        write_array_csv(np.array([[fsc_val]]), f'{folder_results}/fsc.csv')
        write_array_csv(np.array([[temps_calcul]]), f'{folder_results}/temps_calcul.csv')


