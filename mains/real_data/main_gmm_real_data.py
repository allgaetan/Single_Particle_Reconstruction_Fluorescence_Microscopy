from learning_algorithms.gradient_descent_importance_sampling import gd_importance_sampling_3d
from learning_algorithms.gradient_descent_known_angles import gradient_descent_known_rot
from volume_representation.gaussian_mixture_representation.GMM_representation import GMM_representation
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import nd_gaussian, make_grid
from classes_with_parameters import ParametersMainAlg, ParametersGMM
from volume_representation.pixel_representation import Fourier_pixel_representation
from manage_files.read_save_files import read_image, read_images_in_folder, read_csv
from manage_files.paths import *
from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly
import numpy as np
from manage_files.read_save_files import make_dir, save
from common_image_processing_methods.others import crop_center
from learning_algorithms.GMM_coarse_to_fine import coarse_to_fine_gmm_optimization
from classes_with_parameters import ParametersDataGeneration
from common_image_processing_methods.otsu_thresholding import otsu_thresholding

nb_dim = 3
size = 50
nb_gaussians = 100
sigma = 0.03
pth_inter = f'{PATH_PROJECT_FOLDER}/real_data'
pth_views = f'{pth_inter}/c1_selection'
# pth_rot_vecs = f'{PATH_RESULTS_SUMMARY}/real_data/FSC/test_0/intermediar_results/estimated_rot_vecs_epoch_1.csv'
#pth_rot_vecs = (f'{PATH_PROJECT_FOLDER}/results/real_data_centriole_gmm_known_poses/intermediar_results/'
 #               f'estimated_rot_vecs_epoch_6.csv')
#rot_vecs = read_csv(pth_rot_vecs)
# rot_vecs = 180*np.random.random((47,3))
pth_thresholded = f'{pth_inter}/c1_thersholded'
make_dir(pth_thresholded)
images, fns = read_images_in_folder(pth_views)
for i in range(len(images)):
    im = images[i]
    thresholded = otsu_thresholding(im, 0.5)
    save(f'{pth_thresholded}/{fns[i]}', thresholded)


psf_true = read_image(f'{pth_inter}/PSF_6_c1_resized_ratio_2.tif')
save_fold = f'{PATH_PROJECT_FOLDER}/results/real_data_centriole_gmm_selection'
make_dir(save_fold)

fwhm_psf_xy = 50 # width of PSF in xy (nm)
fwhm_psf_z = 210 # width of PSF in z (nm)
pixel_size = 700
sigma_PSF_pixels_xy = fwhm_psf_xy/(np.sqrt(np.log(2))* pixel_size)
sigma_PSF_pixels_z = fwhm_psf_z / (np.sqrt(np.log(2))*pixel_size)
cov_PSF = np.array([[sigma_PSF_pixels_z**2, 0, 0],
                            [0, sigma_PSF_pixels_xy**2, 0],
                            [0, 0, sigma_PSF_pixels_xy**2]])

#params_data_gen = ParametersDataGeneration()
#cov_PSF = params_data_gen.get_cov_psf()
print('cov psf', cov_PSF)
grid = make_grid(size, nb_dim)
psf = nd_gaussian(grid, np.zeros(nb_dim), cov_PSF, nb_dim)
psf_true = crop_center(psf_true, (size, size, size))
save(f'{save_fold}/psf.tif', psf)
save(f'{save_fold}/psf_true.tif', psf_true)

views_thresh, file_names = read_images_in_folder(pth_thresholded)

#volume_representation = Fourier_pixel_representation(3, size, psf_true)


volume_representation = GMM_representation(nb_gaussians, sigma, nb_dim, size, cov_PSF, 0.01)
volume_representation.init_with_average_of_views(views_thresh)
volume_representation.register_and_save(save_fold, 'init_gmm_vol')
#volume_representation.save_gmm_parameters(save_fold)


param_gm = ParametersGMM(nb_gaussians_init=15, sigma_init=0.1, nb_gaussians_ratio=4, sigma_ratio=1.4, nb_steps=4,
                             threshold_gaussians=0.1, unif_prop_mins=[0.5,0.25,0.125,0])

params_learning_alg = ParametersMainAlg(eps=-100, N_iter_max=50, lr=10**-4, N_axes=25, N_rot=20, convention='ZXZ')
true_trans_vecs = np.zeros((len(views_thresh), 3))

uniform_sphere_discretization = discretize_sphere_uniformly(params_learning_alg.M_axes, params_learning_alg.M_rot)
imp_distrs_axes = np.ones((len(views_thresh), params_learning_alg.M_axes)) / params_learning_alg.M_axes
imp_distrs_rot = np.ones((len(views_thresh), params_learning_alg.M_rot)) / params_learning_alg.M_rot

#gradient_descent_known_rot(volume_representation, rot_vecs, true_trans_vecs, views_thresh, params_learning_alg, False, save_fold)


gd_importance_sampling_3d(volume_representation, uniform_sphere_discretization, true_trans_vecs, views_thresh, imp_distrs_axes,
                              imp_distrs_rot, 1,  params_learning_alg.prop_min, params_learning_alg, False, save_fold, use_gpu=False)


volume_representation.register_and_save(save_fold, 'recons')
"""
(volume_representation, _, imp_distrs_axes, imp_distrs_rot,
     recorded_shifts, nb_iter_each_step, time_each_steps, uniform_sphere_discretization) = (
    coarse_to_fine_gmm_optimization(save_fold, views, params_learning_alg, param_gm, cov_PSF, None, true_trans_vecs, False,
                                    False, False, 1,3, 50))
"""
#