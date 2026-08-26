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


"""Main code to apply reconstruction on sythtetic data using gaussian mixture to represent the volume. 
It generates synthetic views from a ground truth image and then used them for the reconstruction

Before running  this code to a specify ground truth image, run the code 'preprocess_ground_truth.py' (located in subfolder 'preprocessing') 
to preprocess the ground truth image if it hasn't already been done"""


def main_ctf_gmm(fold_ground_truth, ground_truth_name, folder_results, folder_views, params_data_gen, params_learning_alg,
                 params_gmm, known_rot=True, known_axes=True, known_trans=False):

    """
    fold_ground_truth : folder containing the ground_truth image from which views are generated
    ground_truth_name : name of the ground_truth image (with extension)
    folder_results : folder in which results are saved
    folder_views : generated_views are saved here
    nb_dim : number of dimensions (2 or 3)
    noise : standard deviation of gaussian noise added to images
    sig_z : standard deviation of PSF along z axis
    nb_views : number of views generated
    sigma_trans_ker : to generate the views, the translations are drawn using a normal distribution of standard deviation
        sigma_trans_ker
    eps : coefficient involved in stopping criteria
    lr : learning rate
    N_iter_max : maximum number of iteration at each coarse to fine step
    N_axes : number of axes drawn by importance sampling (2 first euler angles)
    coeff_kernel_axes : coefficient of the kernel associated to axes
    N_rot : number of in plane rotation drawn by importance sampling (third euler angle)
    coeff_kernel_rot : coefficient of the kernel associated to in plane rotations
    M_axes : number of axes in the discretization
    M_rot : number of in-plane rot in the discretization
    dec_prop : the proportion of uniform distribution is divided by this factor between to epochs
    nb_gaussians_init : init
    nb_gaussians_ratio : The number of gaussians is multiplied by this factor between 2 coarse-to-fine steps
    sigma_init : standard deviation  of gaussians at the first step (proportion of image size)
    sigma_ratio : strandard deviation of gaussians is divided by this factor between two coarse-to-fine steps
    nb_steps : number of coarse-to-fine steps
    threshold_gaussians : gaussians which coefficient becomes  lower than 'threshold_gaussiens' during the training are suppresed
    unif_prop_mins : the proportion of uniform distribution in importance distribution is threshold at each coarse to fine step s
        by unif_prop_mins[s]
    known_rot : if True, in-plane-rotations (third euler angle) are supposed known
    known_axes : if True, axes (represented by two first euler angles) are supposed known
    known_trans : if True, translations are supposed to be known
    """
    generate_and_save_data(folder_views, fold_ground_truth, ground_truth_name, params_data_gen)
    make_dir(folder_results)
    params = locals()
    views, rot_vecs, trans_vecs, file_names = read_views(folder_views, params_data_gen.nb_dim)
    print('rot vecs', rot_vecs)

    f = open(f'{folder_results}/views_order.txt', 'w')
    print(file_names, file=f)
    cov_PSF = params_data_gen.get_cov_psf()
    print('cov psf', cov_PSF)

    volume_representation, recorded_energies, imp_distrs_axes, imp_distrs_rot, recorded_shifts, nb_iter_each_step, time_each_steps, uniform_sphere_discretization = coarse_to_fine_gmm_optimization(
        folder_results, views, params_learning_alg, params_gmm, cov_PSF, rot_vecs, trans_vecs, known_trans, known_axes,
        known_rot, 1, params_data_gen.nb_dim, params_data_gen.size)
    # volume_representation = GMM_representation(nb_gaussians=50, sigma=0.0728, nb_dim=nb_dim, size=size, cov_PSF=cov_PSF, threshold_gaussians=threshold_gaussians)
    # gradient_descent_known_rot(volume_representation, rot_vecs, trans_vecs, views, eps, lr, N_iter_max, unknwon_trans)
    if not known_rot or not known_axes:
        mean_errors_angle, _, angles_found = \
            plot_importance_distributions_2(folder_results, rot_vecs, imp_distrs_axes, imp_distrs_rot,
                                            params_data_gen.nb_dim, None, uniform_sphere_discretization, None, params_data_gen.convention)
    else:
        mean_errors_angle = np.zeros(params_data_gen.nb_views)
    if not known_trans:
        mean_error_true_shift(folder_results, recorded_shifts, trans_vecs, nb_iter_each_step)
        pass
    plot_energies(recorded_energies, folder_results)
    save_time_train_param(folder_results, params, params_data_gen.nb_views, time_each_steps, nb_iter_each_step)
    print(f"results saved at location {folder_results}")
    return mean_errors_angle, volume_representation


def save_time_train_param(results_folder, params, nb_views, time_each_steps, actual_number_iter):
    loc = f'{results_folder}/training_param_time.txt'
    f = open(loc, 'w')
    total_time = np.sum(time_each_steps)
    print('total time', total_time, file=f)
    print('time of each gradient descent step', time_each_steps, file=f)
    for i in range(len(actual_number_iter)):
        if actual_number_iter[i] == 0:
            actual_number_iter[i] = 1
    time_iteration = [time_each_steps[i] / actual_number_iter[i] for i in range(len(actual_number_iter))]
    time_iteration_per_view = [time_iteration[i] / nb_views for i in range(len(time_iteration))]
    print('time epoch', [time_each_steps[i] / actual_number_iter[i] for i in range(len(actual_number_iter))], file=f)
    print('nb iters', nb_views*np.array(actual_number_iter), file=f)
    print('time iteration', time_iteration_per_view, file=f)
    # print('ssim to groud truth', self.ssims[-1], file=f)
    print_dictionnary_in_file(params, f)
    return f


if __name__ == '__main__':
    import os
    from volume_representation.gaussian_mixture_representation.GMM_image_fitting import find_nb_gaussians_given_sigma
    from common_image_processing_methods.others import *
    fold_gt = "../../../ground_truths"
    """
    gt = 'npc_im_0'
    gt_im = read_image(f'{fold_gt}/{gt}.tif')
    gt_padded = crop_center(gt_im, (70,70,70))
    save(f'{fold_gt}/{gt}_padded.tif', gt_padded)
    sig = 0.07
    params_learning_alg = ParametersMainAlg(lr=10 ** -4, N_iter_max=50, eps=-5)
    params_data_gen = ParametersDataGeneration()
    cov_PSF = params_data_gen.get_cov_psf()
    params_gmm = ParametersGMM(nb_gaussians_init=150, nb_steps=1, sigma_init=sig, init_with_views=True,
                               threshold_gaussians=0.01)
    folder_results = f'{PTH_LOCAL_RESULTS}/gmm_test/{gt}/init_with_avg_of_views'
    folder_views = f'{folder_results}/views'
    _, volume_representation = main_ctf_gmm(fold_gt, f'{gt}_padded.tif', folder_results, folder_views,
                                            params_data_gen, params_learning_alg,
                                            params_gmm, known_rot=False, known_axes=False, known_trans=False)

    """
    gts = [ "emd_0680", "HIV-1-Vaccine_prep"]

    #sigma_gaussians = [0.1, 0.08, 0.07, 0.05, 0.03]
    sigma_gaussians = [0.03,0.05,0.07,0.08,0.1,0.12]
    convention = 'ZXZ'
    params_data_gen = ParametersDataGeneration(nb_views=10, convention=convention)
    params_learning_alg = ParametersMainAlg(lr=2*10 ** -4, N_iter_max=50, eps=0, convention=convention)
    cov_PSF = params_data_gen.get_cov_psf()
    nb_tests_per_points = 10
    size = 50
    for ground_truth_name in gts:
        #nb_gaussians = [2,5,10,15,20,25,30,40,50,75,100,125,150,175,200,250]
        nb_gaussians = [2,5,10,15,20,25,30,40,50,75,100,125,150,175,200,250]
        ssims = np.zeros((len(sigma_gaussians), len(nb_gaussians), nb_tests_per_points))
        fscs = np.zeros((len(sigma_gaussians), len(nb_gaussians), nb_tests_per_points))
        fold_root_0 = f'{PTH_LOCAL_RESULTS}/gmm_test_nb_gaussians_2/{ground_truth_name}/init_with_avg_of_views'
        for i,sig in enumerate(sigma_gaussians):
            fold_root = f'{fold_root_0}/sig_{sig}'
            make_dir(fold_root)
            gt = read_image(f'{fold_gt}/{ground_truth_name}.tif')
            views, _, _, _, _, _, _ = generate_data(gt, params_data_gen)
            cov = sig ** 2 * np.eye(params_data_gen.nb_dim) + cov_PSF
            avg_opt_nb_gauss = 0
            for k in range(len(views)):
                view = views[k]
                nb_gauss = find_nb_gaussians_given_sigma(view, sig ** 2 * np.eye(params_data_gen.nb_dim), size)
                avg_opt_nb_gauss += nb_gauss
            avg_opt_nb_gauss /= len(views)
            print('opt nb gauss', avg_opt_nb_gauss)
            write_array_csv(np.array([[avg_opt_nb_gauss]]), f'{fold_root}/opt_nb_gaussians.csv')

            for j,nb_gauss in enumerate(nb_gaussians):
                print('nb gaussians', nb_gauss)
                params_gmm = ParametersGMM(nb_gaussians_init=nb_gauss, nb_steps=1, sigma_init=sig, init_with_views=True, threshold_gaussians=0.01)
                for t in range(nb_tests_per_points):
                    folder_results = f'{fold_root}/nb_gaussians_{nb_gauss}/test_{t}'
                    folder_views = f'{folder_results}/views'
                    gt_im = generate_and_save_data(folder_views, fold_gt, f'{ground_truth_name}.tif', params_data_gen)
                    make_dir(folder_results)
                    params = locals()
                    views, rot_vecs, trans_vecs, file_names = read_views(folder_views, params_data_gen.nb_dim)
                    gmm_rep = GMM_representation(params_gmm.nb_gaussians_init, params_gmm.sigma_init, 3, size,
                                                 cov_PSF, params_gmm.threshold_gaussians)
                    if params_gmm.init_with_views:
                        gmm_rep.init_with_average_of_views(views)
                    gmm_rep.register_and_save(folder_results, 'init_gmm_vol')
                    gmm_rep.save_gmm_parameters(folder_results)
                    gmm_rep_out, energies, _, _ = gradient_descent_known_rot(gmm_rep, rot_vecs, trans_vecs, views, params_learning_alg, True,
                                                                             save_fold=folder_results)
                    gm = gmm_rep_out.evaluate_on_grid()
                    write_array_csv(energies, f'{folder_results}/energies.csv')
                    save(f'{folder_results}/recons.tif', gm)
                    ssim_val = ssim(gm, gt)
                    fsc_val = fsc(gm, gt, cutoff=0.143)
                    ssims[i, j, t] = ssim_val
                    fscs[i, j, t] = fsc_val
                    write_array_csv(np.array([[ssim_val]]), f'{folder_results}/ssim.csv')
                    write_array_csv(np.array([[fsc_val]]), f'{folder_results}/fsc.csv')
                    
            write_array_csv(ssims[i], f'{fold_root}/ssims.csv')
            write_array_csv(fscs[i], f'{fold_root}/fscs.csv')

        from manage_matplotlib.graph_setup import *
        labels = [f'sigma = {sig/2}' for sig in sigma_gaussians]
        plot_experiment_graph(nb_gaussians, ssims, 'number of gaussians', 'ssim', '', labels)
        plt.savefig(f'{fold_root_0}/ssim_wrt_nb_gauss.png')
        plt.close()

        from manage_matplotlib.graph_setup import *
        plot_experiment_graph(nb_gaussians, fscs, 'number of gaussians', 'Fourier Shell Correlation', '', labels)
        plt.savefig(f'{fold_root_0}/fsc_wrt_nb_gauss.png')
        plt.close()














