from volume_representation.gaussian_mixture_representation.GMM_representation import GMM_representation
from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly
from learning_algorithms.gradient_descent_importance_sampling import gd_importance_sampling_3d, gradient_descent_importance_sampling_known_axes,\
    gradient_descent_importance_sampling_known_rot
import numpy as np
import copy as cp
from learning_algorithms.gradient_descent_known_angles import gradient_descent_known_rot
from time import time


def coarse_to_fine_gmm_optimization(folder_results, views, params_learning_alg, params_gmm, cov_PSF, true_rot_vecs, true_trans_vecs, known_trans, known_axes,
        known_rot, init_unif_prop, nb_dim, size):

    if isinstance(params_learning_alg.N_iter_max, int):
        N_iter_maxs = [params_learning_alg.N_iter_max for _ in range(params_gmm.nb_steps)]
    unif_prop = init_unif_prop
    """
    if params_learning_alg.prop_min is None:
        unif_prop_mins = np.zeros(params_gmm.nb_steps)
    elif isinstance(params_learning_alg.prop_min, float):
        unif_prop_mins = [params_learning_alg.prop_min] * params_gmm.nb_steps
    else:
        print('cc')
        unif_prop_mins = params_learning_alg.prop_min
    """
    unif_prop_mins = [0]*params_gmm.nb_steps
    gmm_rep = GMM_representation(params_gmm.nb_gaussians_init, params_gmm.sigma_init, nb_dim, size, cov_PSF, params_gmm.threshold_gaussians)
    if params_gmm.init_with_views:
        gmm_rep.init_with_average_of_views(views)
    gmm_rep.register_and_save(folder_results, 'init_gmm_vol')
    uniform_sphere_discretization = discretize_sphere_uniformly(params_learning_alg.M_axes, params_learning_alg.M_rot)
    imp_distrs_axes = np.ones((len(views), params_learning_alg.M_axes)) / params_learning_alg.M_axes
    imp_distrs_rot = np.ones((len(views), params_learning_alg.M_rot)) / params_learning_alg.M_rot
    imp_distrs_rot_recorded_all_steps = []
    imp_distrs_axes_recorded_all_steps = []
    recorded_energies_all_steps = []
    recorded_shifts_all_steps = [[] for _ in range(len(views))]
    nb_iter_each_step = []
    time_each_steps = np.zeros(params_gmm.nb_steps)
    for step in range(params_gmm.nb_steps):
        print('number of gaussians', len(gmm_rep.centers))
        t = time()
        if known_axes and known_rot:
            gmm_rep, recorded_energies, recorded_shifts, itr = gradient_descent_known_rot(gmm_rep, true_rot_vecs, true_trans_vecs, views, params_learning_alg, known_trans)

        elif known_axes and not known_rot:
            imp_distrs_rot_recorded, recorded_energies, recorded_shifts, unif_prop, gmm_rep, itr = gradient_descent_importance_sampling_known_axes(gmm_rep, uniform_sphere_discretization, true_rot_vecs, true_trans_vecs, views,
                                                   imp_distrs_rot, unif_prop, unif_prop_mins[step], params_learning_alg, known_trans
                                                            )
            imp_distrs_rot = imp_distrs_rot_recorded[-1]
            imp_distrs_rot_recorded_all_steps += cp.deepcopy(imp_distrs_rot_recorded)
        elif known_rot and not known_axes:
            imp_distrs_axes_recorded, recorded_energies, recorded_shifts, unif_prop, gmm_rep, itr = gradient_descent_importance_sampling_known_rot(
                gmm_rep,
                uniform_sphere_discretization, true_rot_vecs, true_trans_vecs, views,
                imp_distrs_axes, unif_prop, unif_prop_mins[step], params_learning_alg, known_trans)
            imp_distrs_axes = imp_distrs_axes_recorded[-1]
            imp_distrs_axes_recorded_all_steps += cp.deepcopy(imp_distrs_axes_recorded)
        else:
            imp_distrs_rot_recorded, imp_distrs_axes_recorded, recorded_energies, recorded_shifts, unif_prop, gmm_rep, itr, _, _, _, _ = \
                gd_importance_sampling_3d(gmm_rep, uniform_sphere_discretization,
                                      true_trans_vecs, views, imp_distrs_axes,
                                      imp_distrs_rot, unif_prop, unif_prop_mins[step], params_learning_alg, known_trans, folder_results)
            imp_distrs_rot = imp_distrs_rot_recorded[-1]
            imp_distrs_axes = imp_distrs_axes_recorded[-1]
            imp_distrs_rot_recorded_all_steps += cp.deepcopy(imp_distrs_rot_recorded)
            imp_distrs_axes_recorded_all_steps += cp.deepcopy(imp_distrs_axes_recorded)
        t_step = time()-t
        time_each_steps[step] = t_step
        gmm_rep.register_and_save(folder_results, f'step_{step}', ground_truth_path=None)
        if step != params_gmm.nb_steps-1:
            print('nb gauss ratio', params_gmm.nb_gaussians_ratio)
            gmm_rep.split_gaussians(params_gmm.nb_gaussians_ratio, params_gmm.sigma_ratio)
        recorded_energies_all_steps += recorded_energies
        for v in range(len(views)):
            recorded_shifts_all_steps[v] += recorded_shifts[v]
        nb_iter_each_step.append(itr)
        gmm_rep.save_gmm_parameters(folder_results)
    return gmm_rep, recorded_energies_all_steps, imp_distrs_axes_recorded_all_steps, imp_distrs_rot_recorded_all_steps, \
           recorded_shifts_all_steps, nb_iter_each_step, time_each_steps, uniform_sphere_discretization





