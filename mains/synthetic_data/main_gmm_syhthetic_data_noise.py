import matplotlib.pyplot as plt
from data_generation.generate_data import generate_and_save_data, read_views, generate_data
from learning_algorithms.GMM_coarse_to_fine import coarse_to_fine_gmm_optimization
import numpy as np
from metrics_and_visualisation.error_orientation_translation import plot_importance_distributions_2, plot_energies
from manage_files.read_save_files import *
from metrics_and_visualisation.error_orientation_translation import mean_error_true_shift
from manage_files.paths import PTH_LOCAL_RESULTS
from classes_with_parameters import *
from metrics_and_visualisation.metrics_to_compare_2_images import *
from manage_matplotlib.plot_graph import plot_experiment_graph
from volume_representation.gaussian_mixture_representation.GMM_representation import GMM_representation
from learning_algorithms.gradient_descent_known_angles import gradient_descent_known_rot
import os
from volume_representation.gaussian_mixture_representation.GMM_image_fitting import find_nb_gaussians_given_sigma
from common_image_processing_methods.others import *
fold_gt = "../../../ground_truths"

#sigma_gaussians = [0.1, 0.08, 0.07, 0.05, 0.03]
convention = 'ZXZ'
params_data_gen = ParametersDataGeneration(nb_views=20, convention=convention, snr=0.1)
params_learning_alg = ParametersMainAlg(lr=2*10 ** -4, N_iter_max=30, eps=0, convention=convention)
cov_PSF = params_data_gen.get_cov_psf()
print('cov psf', cov_PSF)
size = 50
sig = 0.03
nb_gauss = 250
ground_truth_name = "recepteurs_AMPA"
#nb_gaussians = [2,5,10,15,20,25,30,40,50,75,100,125,150,175,200,250]
nb_gaussians = [250]
folder_results = f'{PTH_LOCAL_RESULTS}/gmm_test_noise/{ground_truth_name}'
make_dir(folder_results)
gt = read_image(f'{fold_gt}/{ground_truth_name}.tif')
views, rot_vecs, trans_vecs, _, _, _, _ = generate_data(gt, params_data_gen)
cov = sig ** 2 * np.eye(params_data_gen.nb_dim) + cov_PSF
params_gmm = ParametersGMM(nb_gaussians_init=nb_gauss, nb_steps=1, sigma_init=sig, init_with_views=True, threshold_gaussians=0.01)
folder_views = f'{folder_results}/views'
gt_im = generate_and_save_data(folder_views, fold_gt, f'{ground_truth_name}.tif', params_data_gen)
make_dir(folder_results)
params = locals()
gmm_rep = GMM_representation(params_gmm.nb_gaussians_init, params_gmm.sigma_init, 3, size,
                             cov_PSF, params_gmm.threshold_gaussians)
if params_gmm.init_with_views:
    gmm_rep.init_with_average_of_views(views)
gmm_rep.register_and_save(folder_results, 'init_gmm_vol')
gmm_rep.save_gmm_parameters(folder_results)
gmm_rep_out, energies, _, _ = gradient_descent_known_rot(gmm_rep, rot_vecs, trans_vecs, views, params_learning_alg, True, folder_results)
gm = gmm_rep_out.evaluate_on_grid()
write_array_csv(energies, f'{folder_results}/energies.csv')
save(f'{folder_results}/recons.tif', gm)
ssim_val = ssim(gm, gt)
fsc_val = fsc(gm, gt, cutoff=0.143)



