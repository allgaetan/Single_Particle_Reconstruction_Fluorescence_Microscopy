import os
from skimage import io
#from NPC.NPC_c8 import read_image
from metrics_and_visualisation.metrics_to_compare_2_images import fsc
from data_generation.generate_data import generate_and_save_data, read_views, get_PSF_with_stds
from volume_representation.pixel_representation import Fourier_pixel_representation
from learning_algorithms.gradient_descent_known_angles import gradient_descent_known_rot
from learning_algorithms.gradient_descent_importance_sampling import gradient_descent_importance_sampling_known_axes, \
    gradient_descent_importance_sampling_known_rot, gd_importance_sampling_3d
from manage_files.read_save_files import make_dir, print_dictionnary_in_file, write_array_csv, delete_dir
from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly
from metrics_and_visualisation.error_orientation_translation import plot_importance_distributions_2, plot_energies
from time import time
from common_image_processing_methods.others import normalize
from skimage.metrics import structural_similarity as ssim
from common_image_processing_methods.rotation_translation import get_rotation_matrix
from manage_files.read_save_files import save, read_images_in_folder
from metrics_and_visualisation.metrics_to_compare_2_images import f1_score
from classes_with_parameters import *
from manage_files.paths import *

"""Main code to apply reconstruction on sythtetic data using pixel representation in Fourier domain to represent the volume. 
It generates synthetic views from a ground truth image and then used them for the reconstruction

Before running  this code to a specify ground truth image, run the code 'preprocess_ground_truth.py' (located in subfolder 'preprocessing') 
to preprocess the ground truth image if it hasn't already been done"""


def main_pixel_representation(fold_ground_truth, ground_truth_name, folder_results, folder_views, params_data_gen, params_learning_alg,
                              nb_dim=3, known_axes=False, known_rot=False, known_trans=False, symmetry_rot_mats=None, output=None,
                              delete_views=False, use_gpu=False):

    """
    fold_ground_truth : folder containing the ground_truth image from which views are generated
    ground_truth_name : name of the ground_truth image
    folder_results : folder in which results are saved
    nb_dim : number of dimensions (2 or 3)
    snr : signal to noise ratio
    sig_z : standard deviation of PSF along z axis
    nb_views : number of views generated
    sigma_trans_ker : to generate the views, the translations are drawn using a normal distribution of standard deviation
        sigma_trans_ker
    size : size of images
    eps : coefficient involved in stopping criteria
    lr : learning rate
    N_iter_max : maximum number of iteration
    N_axes : number of axes drawn by importance sampling
    coeff_kernel_axes : coefficient of the kernel associated to axes
    N_rot : number of in plane rotation drawn by importance sampling (third euler angle)
    coeff_kernel_rot : coefficient of the kernel associated to in plane rotations
    M_axes : number of axes in the discretization
    M_rot : number of in-plane rot in the discretization
    dec_prop : the proportion of uniform distribution is divided by this factor between to epochs
    known_rot : if True, in-plane-rotations (third euler angle) are supposed known
    known_axes : if True, axes (represented by two first euler angles) are supposed known
    known_trans : if True, translations are supposed to be known
    """
    make_dir(folder_results)
    if symmetry_rot_mats is None:
        if ground_truth_name == "synth_centriole_prep.tif":
            symmetry_rot_mats = [get_rotation_matrix([0, 360 * k / 9, 0], params_data_gen.convention) for k in range(9)]
        if ground_truth_name == "emd_0680.tif":
            symmetry_rot_mats = [get_rotation_matrix([0, 360 * k / 5, 0], params_data_gen.convention) for k in range(5)]

    params_data_gen_vals = params_data_gen.params
    params_learning_alg_vals = params_learning_alg.params
    PSF = params_data_gen.get_psf()
    """
    supp_zero_idxw = PSF>0.01
    ma = np.max(PSF)
    a = 1
    PSF = np.zeros((size, size, size))
    PSF[size//2, size//2, size//2] = 1
    """
    save(f'{folder_results}/PSF.tif', PSF)
    thetas_unif, phis_unifs, psis_unif = discretize_sphere_uniformly(20, 20)
    thetas_unif += np.random.normal(0, 3, thetas_unif.shape)
    phis_unifs += np.random.normal(0, 3, phis_unifs.shape)
    psis_unif += np.random.normal(0, 3 , psis_unif.shape)
    rot_vecs = np.array([thetas_unif, phis_unifs, psis_unif]).T
    ground_truth = generate_and_save_data(folder_views, fold_ground_truth, ground_truth_name, params_data_gen)
    views, rot_vecs, trans_vecs, file_names = read_views(folder_views, nb_dim)
    """
    views, file_names = read_images_in_folder(folder_views, alphabetic_order=False)
    rot_vecs = []
    # rot_mats = []
    nb_views = len(file_names)
    for v in range(nb_views):
        _, dil_val, rv1, rv2, rv3, _ = file_names[v].split('_')
        rot_vec = [float(rv1), float(rv2), float(rv3)]
        rot_vecs.append(rot_vec)
        # rot_mat = get_3d_rotation_matrix(rot_vec, convention='ZXZ')
        # rot_mats.append(rot_mat)

    # rot_mats = np.array(rot_mats)
    rot_vecs = np.array(rot_vecs)
    trans_vecs = np.zeros((nb_views, 3))
    """
    """
    d = f'/home/eloy/exemple_vues_PSF/{ground_truth_name}/stdz{sig_z}_stdxy{sig_xy}'
    make_dir(d)
    save(f'{d}/PSF.tif', PSF)
    for v in range(nb_views):
        save(f'{d}/{file_names[v]}', views[v])
        
    """
    #init_vol = np.mean(np.array(views), axis=0)
    init_vol = None
    fourier_pixel_rep = Fourier_pixel_representation(params_data_gen.nb_dim, params_data_gen.size, PSF, random_init=False, init_vol=init_vol)
    uniform_sphere_discretization = discretize_sphere_uniformly(params_learning_alg.M_axes, params_learning_alg.M_rot)
    imp_distrs_rot = np.ones((len(views), params_learning_alg.M_rot)) / params_learning_alg.M_rot
    imp_distrs_axes = np.ones((len(views), params_learning_alg.M_axes)) / params_learning_alg.M_axes
    f = open(f'{folder_results}/views_order.txt', 'w')
    print(file_names, file=f)
    t = time()
    if known_axes and not known_rot:

        imp_distrs_rot_recorded, recorded_energies, recorded_shifts, unif_prop, volume_representation, itr = \
            gradient_descent_importance_sampling_known_axes(fourier_pixel_rep, uniform_sphere_discretization, rot_vecs, trans_vecs, views,
                                                       imp_distrs_rot, 1, params_learning_alg.prop_min, params_learning_alg, known_trans)
        imp_distrs_axes_recorded = []
    elif known_rot and not known_axes:
        imp_distrs_axes_recorded, recorded_energies, recorded_shifts, unif_prop, volume_representation, itr = \
            gradient_descent_importance_sampling_known_rot(fourier_pixel_rep, uniform_sphere_discretization, rot_vecs, trans_vecs, views,
                                                       imp_distrs_axes, 1, params_learning_alg.prop_min, params_learning_alg, known_trans)
        imp_distrs_rot_recorded = []
    elif not known_axes and not known_rot:

        imp_distrs_rot_recorded, imp_distrs_axes_recorded, recorded_energies, recorded_shifts, unif_prop, volume_representation, itr,\
            _, _, _, _ = \
            gd_importance_sampling_3d(fourier_pixel_rep, uniform_sphere_discretization, trans_vecs, views,
                                  imp_distrs_axes,
                                  imp_distrs_rot, 1, params_learning_alg.prop_min, params_learning_alg, known_trans, folder_results, ground_truth_path=None, use_gpu=use_gpu)
    else:
        volume_representation, recorded_energies, recorded_shifts, itr = gradient_descent_known_rot(fourier_pixel_rep,
                                   rot_vecs, trans_vecs,
                                   views, params_learning_alg, known_trans, folder_results)
        imp_distrs_axes_recorded, imp_distrs_rot_recorded = [], []
    total_time = time() - t
    plot_energies(recorded_energies, folder_results)

    if not known_rot or not known_axes:
        mean_errors_angle, _, angles_found = \
            plot_importance_distributions_2(folder_results, rot_vecs, imp_distrs_axes_recorded, imp_distrs_rot_recorded, params_data_gen.nb_dim, None, uniform_sphere_discretization,
                                            symmetry_rot_mats, params_learning_alg.convention)

    if not known_axes or not known_rot:
        ground_truth_path = f'{fold_ground_truth}/{ground_truth_name}'
        #ground_truth_path = None
        im = fourier_pixel_rep.register_and_save(folder_results, 'recons', ground_truth_path=ground_truth_path, translate=not known_trans)
    else:
        im = fourier_pixel_rep.save(folder_results, 'recons')
    from skimage.restoration import denoise_tv_chambolle
    denoised_im = denoise_tv_chambolle(im, weight=0.05)
    save(f'{folder_results}/recons_denoised.tif', denoised_im)
    loc = f'{folder_results}/training_param_time.txt'
    f = open(loc, 'w')
    print_dictionnary_in_file(params_data_gen_vals, f)
    print_dictionnary_in_file(params_learning_alg_vals, f)
    print(f'total time : {total_time}', file=f)
    print(f'total time : {total_time}', file=f)
    print(f"results saved at location {folder_results}")

    ssim_to_gt = ssim(im, ground_truth)
    fsc_to_gt = fsc(im, ground_truth)
    fsc_to_gt_normalized = fsc(normalize(im), ground_truth)
    f1_score_val = f1_score(im, ground_truth)
    write_array_csv(np.array([[ssim_to_gt]]), f'{folder_results}/ssim.csv')
    write_array_csv(np.array([[fsc_to_gt]]), f'{folder_results}/fsc.csv')
    write_array_csv(np.array([[fsc_to_gt_normalized]]), f'{folder_results}/fsc_normalized.csv')
    # plot_conical_fsc(im, ground_truth, folder_results)
    if delete_views:
        delete_dir(folder_views)
    save(f'{folder_results}/ground_truth.tif', ground_truth)
    #radiuses_frequencies, thetas, phis, conical_fsc = plot_conical_fsc(ground_truth, im, folder_results)
    return ssim_to_gt, f1_score_val, im







"""
main_pixel_representation(fold_gt, f'{ground_truth_name}.tif', folder_results, folder_views,
                          known_axes=True, known_rot=False, known_trans=True, lr=0.1, sigma_trans_ker=0
                          ,dec_prop=1.15)

main_pixel_representation(fold_gt, f'{ground_truth_name}.tif', f'{folder_results}_0', folder_views,
                          known_axes=False, known_rot=True, known_trans=True, lr=0.1, sigma_trans_ker=0
                          ,dec_prop=1.15)

main_pixel_representation(fold_gt, f'{ground_truth_name}.tif', f'{folder_results}_1', folder_views,
                          known_axes=False, known_rot=True, known_trans=True, lr=0.1, sigma_trans_ker=3
                          ,dec_prop=1.15)

main_pixel_representation(fold_gt, f'{ground_truth_name}.tif', f'{folder_results}_2', folder_views,
                          known_axes=False, known_rot=True, known_trans=False, lr=0.1, sigma_trans_ker=3
                          ,dec_prop=1.15)
"""
if __name__ == '__main__':
    """
    from manage_files.paths import *
    import argparse
    parser = argparse.ArgumentParser()
    from inspect import signature
    sign = signature(main_pixel_representation)
    params = list(sign.parameters.keys())
    import numpy as np
    """
    from manage_files.read_save_files import read_image
    pth = "/home/eloy/Téléchargements/archive/isotropic_downsampled"
    views, _ = read_images_in_folder(pth)
    pth_psf = "/home/eloy/Téléchargements/archive"
    PSF = read_image(f'{pth_psf}/PSFModel-0.ome.tiff')
    trans_vecs = np.zeros((len(views), 3))
    size = 25
    params_data_gen = ParametersDataGeneration(nb_views=10, sig_z=5, partial_labelling=False, sigma_trans_ker=0,
                                               size=size, snr=10000)
    params_learning_alg = ParametersMainAlg(N_iter_max=100, convention='ZXZ', eps=-1000000, dec_prop=1.2, lr=0.1)

    fourier_pixel_rep = Fourier_pixel_representation(3, size, PSF,
                                                     random_init=False)
    uniform_sphere_discretization = discretize_sphere_uniformly(params_learning_alg.M_axes, params_learning_alg.M_rot)
    imp_distrs_rot = np.ones((len(views), params_learning_alg.M_rot)) / params_learning_alg.M_rot
    imp_distrs_axes = np.ones((len(views), params_learning_alg.M_axes)) / params_learning_alg.M_axes

    folder_results = "/home/eloy/results_jean_3"
    gd_importance_sampling_3d(fourier_pixel_rep, uniform_sphere_discretization, trans_vecs, views,
                              imp_distrs_axes,
                              imp_distrs_rot, 1, params_learning_alg.prop_min, params_learning_alg, False,
                              folder_results, ground_truth_path=None, use_gpu=False)
    1/0



    # np.random.seed(0)
    fold_gt = PTH_GT
    """
    for ground_truth_name in ["recepteurs_AMPA"]: #, "clathrine", "emd_0680", "synth_centriole_prep", "HIV-1-Vaccine_prep"]:
        for nb_views in range(2,11):
            folder_results = f"../../../results/{ground_truth_name}/nb_views_{nb_views}"
            folder_views = f"../../../results/views/{ground_truth_name}_unknown_trans"
            main_pixel_representation(fold_gt, f'{ground_truth_name}.tif', folder_results, folder_views,
                                  known_axes=False, known_rot=False, known_trans=False, lr=0.1, sigma_trans_ker=0
                                  ,dec_prop=1.15, N_rot=30, N_axes=30, nb_views=nb_views, N_iter_max=300//nb_views)

    """
    n_test = 7
    #psf = io.imread(f'{PATH_REAL_DATA}/Data_marine_raw/PSF/PSF_6_c1.tif')
    from common_image_processing_methods.rotation_translation import rotation
    from common_image_processing_methods.others import crop_center, resize
    snr = 0.01
    N_iters = [150,101,90,80, 70, 60, 60, 60,50,50,50,50,50]
    dec_props = [1.01,1.02]
    for ground_truth_name in ["recepteurs_AMPA"]: #, "clathrine", , "emd_0680"]:
        ssim_avg = 0
        f1_score_avg = 0
        for nb_views in [20]:
            for test in range(10):
                for i,snr in enumerate([0.01,0.02,0.05,0.07,0.1,0.2,0.5,0.7,1]):
                    # ground_truth_name = "HIV-1-Vaccine_prep"
                    params_data_gen = ParametersDataGeneration(nb_views=20, sig_z=5, partial_labelling=False, sigma_trans_ker=0, size=50, snr=snr)
                    params_learning_alg = ParametersMainAlg(N_iter_max=40, convention='ZXZ', eps=-1000000, dec_prop=1.07, lr=0.05)
                    folder_results = f"{PATH_RESULTS_SUMMARY}/test_snr_unlnown_poses/snr_{snr}/test_{test}"
                    #folder_views = f"{PATH_PROJECT_FOLDER}/results_deep_learning/heterogeneity_centriole/views/" \
                     #              f"no_het_s_45_anis_3_alpha_1_nb_views_20_rot_180"
                    folder_views = f'{folder_results}/views'
                    # symmetry_rot_mats = [get_rotation_matrix([0, 360 * k / 5, 0]) for k in range(5)]
                    symmetry_rot_mats = None
                    t = time()
                    # np.random.seed(0)
                    main_pixel_representation(fold_gt, f'{ground_truth_name}.tif', folder_results, folder_views,
                                              params_data_gen, params_learning_alg, known_axes=False, known_rot=False, known_trans=True, use_gpu=True)

                    print('temps', time()-t)

                f1_score_avg/=n_test
                ssim_avg/=n_test
                print(f'{ground_truth_name}, f1 score : {f1_score_avg/n_test}. ssim : {ssim_avg/n_test}')



    # import mains.synthetic_data.main_GMM_synthetic_data




