import matplotlib.pyplot as plt
from data_generation.generate_data import generate_and_save_data, read_views, generate_data
from learning_algorithms.GMM_coarse_to_fine import coarse_to_fine_gmm_optimization
import numpy as np
from metrics_and_visualisation.error_orientation_translation import plot_importance_distributions_2, plot_energies
from manage_files.read_save_files import *
from metrics_and_visualisation.error_orientation_translation import mean_error_true_shift
from manage_files.paths import *
from classes_with_parameters import *
from metrics_and_visualisation.metrics_to_compare_2_images import *
from manage_matplotlib.plot_graph import plot_experiment_graph
from volume_representation.gaussian_mixture_representation.GMM_representation import GMM_rep_volume_and_views
from learning_algorithms.gradient_descent_known_angles import gradient_descent_known_rot
from learning_algorithms.gradient_descent_importance_sampling import gd_importance_sampling_3d
from volume_representation.gaussian_mixture_representation.GMM_image_fitting import Gmm_rep_of_view
from metrics_and_visualisation.plot_conical_fsc import plot_conical_fsc
from time import time


nb_gauss = 250
size = 50
sig = 0.03
# ground_truth_name = 'recepteurs_AMPA'
convention = 'ZXZ'
params_data_gen = ParametersDataGeneration(nb_views=10, convention=convention)
params_learning_alg = ParametersMainAlg(lr=0.05, N_iter_max=50, eps=-20, convention=convention, dec_prop=1.05)
folder_root = f'{PATH_PROJECT_FOLDER}/results_summary/gmm_views_test_nb_gauss'
params_gmm = ParametersGMM(nb_gaussians_init=nb_gauss, nb_steps=1, sigma_init=sig, init_with_views=True, threshold_gaussians=0)

for t in range(10):
    for nb_gaussians_views in [100]:
    #for nb_gaussians_views in np.arange(10, 100, 5):
        for ground_truth_name in ["clathrine"]: # , "clathrine", "emd_0680", "HIV-1-Vaccine_prep"]:
            folder_results = f'{folder_root}/{ground_truth_name}/nb_gauss_{nb_gaussians_views}/test_{t}'
            folder_views = f'{folder_results}/views'
            gt_im = generate_and_save_data(folder_views, PTH_GT, f'{ground_truth_name}.tif', params_data_gen)
            make_dir(folder_results)
            gt = read_image(f'{PTH_GT}/{ground_truth_name}.tif')
            cov_PSF = params_data_gen.get_cov_psf()
            params = locals()
            views, rot_vecs, trans_vecs, file_names = read_views(folder_views, params_data_gen.nb_dim)

            gmm_rep = GMM_rep_volume_and_views(params_gmm.nb_gaussians_init, params_gmm.sigma_init, 3, size,
                                         cov_PSF, params_gmm.threshold_gaussians)

            views_gmm = []
            for i in range(len(views)):
                view_gmm = Gmm_rep_of_view(views[i], sigma=sig, nb_gaussians=nb_gaussians_views)
                views_gmm.append(view_gmm)
                view_gmm.save(f'{folder_views}/view_gmm_{i}.tif')

            if params_gmm.init_with_views:
                gmm_rep.init_with_average_of_views(views)

            """
            gmm_rep.register_and_save(folder_results, 'init_gmm_vol')
            gmm_rep.save_gmm_parameters(folder_results)
            gmm_rep_out, energies, _, _ = gradient_descent_known_rot(gmm_rep, rot_vecs, trans_vecs, views_gmm, params_learning_alg, True, folder_results)
            gm = gmm_rep_out.evaluate_on_grid()
            write_array_csv(energies, f'{folder_results}/energies.csv')
            save(f'{folder_results}/recons.tif', gm)
            ssim_val = ssim(gm, gt)
            fsc_val = fsc(gm, gt, cutoff=0.143)
            gmm_rep.save_gmm_parameters(folder_results)
            """

            uniform_sphere_discretization = discretize_sphere_uniformly(params_learning_alg.M_axes,
                                                                        params_learning_alg.M_rot)
            imp_distrs_axes = np.ones((len(views), params_learning_alg.M_axes)) / params_learning_alg.M_axes
            imp_distrs_rot = np.ones((len(views), params_learning_alg.M_rot)) / params_learning_alg.M_rot
            t0 = time()
            imp_distrs_rot_recorded, imp_distrs_axes_recorded, recorded_energies, recorded_shifts, unif_prop, gmm_rep, itr, _, _, _, _ = \
                gd_importance_sampling_3d(gmm_rep, uniform_sphere_discretization,
                                          trans_vecs, views_gmm, imp_distrs_axes,
                                          imp_distrs_rot, 1, 0, params_learning_alg, True, folder_results)
            temps_calcul = time() - t0
            gmm_rep.save_gmm_parameters(folder_results)
            recons_voxel = gmm_rep.register_and_save(folder_results, 'recons', ground_truth_path=f'{PTH_GT}/{ground_truth_name}.tif')
            mean_errors_angle, _, angles_found = \
                plot_importance_distributions_2(folder_results, rot_vecs, imp_distrs_axes_recorded, imp_distrs_rot_recorded,
                                                params_data_gen.nb_dim, None, uniform_sphere_discretization, None,
                                                params_data_gen.convention)

            plot_energies(recorded_energies, folder_results)
            write_array_csv(recorded_energies, f'{folder_results}/energies.csv')
            write_array_csv(mean_errors_angle, f'{folder_results}/error_angles.csv')
            print('gmm rep shape', recons_voxel.shape)
            print('gt im shape', gt_im.shape)
            plot_conical_fsc(recons_voxel, gt_im, f'{folder_results}/cfsc')
            ssim_val = ssim(recons_voxel, gt_im)
            fsc_val = fsc(recons_voxel, gt_im, cutoff=0.143)
            write_array_csv(np.array([[ssim_val]]), f'{folder_results}/ssim.csv')
            write_array_csv(np.array([[fsc_val]]), f'{folder_results}/fsc.csv')
            write_array_csv(np.array([[temps_calcul]]), f'{folder_results}/temps_calcul.csv')