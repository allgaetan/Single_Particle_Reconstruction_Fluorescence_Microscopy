import time

from common_image_processing_methods.rotation_translation import conversion_2_first_eulers_angles_cartesian, get_rotation_matrix
import numpy as np
#import cupy as cp
from common_image_processing_methods.others import normalize
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import one_d_gaussian
import copy
from learning_algorithms.gradient_descent_known_angles import stopping_criteria
from skimage import io
from manage_files.read_save_files import write_array_csv, make_dir, read_image, save
from skimage.metrics import structural_similarity as ssim


def gradient_descent_importance_sampling_known_axes(volume_representation, uniform_sphere_discretization, true_rot_vecs, true_trans_vecs, views,
                                                   imp_distrs_rot, unif_prop,unif_prop_min, params_learning_alg, known_trans):
    thetas, phis, psis = uniform_sphere_discretization
    M_rot = len(psis)
    imp_distrs_rot_recorded = []
    itr = 0
    recorded_energies = []
    nb_views = len(views)
    recorded_shifts = [[] for _ in range(nb_views)]
    views_fft = [np.fft.fftn(v) for v in views]
    while itr < params_learning_alg.N_iter_max and (not stopping_criteria(recorded_energies, params_learning_alg.eps)):
        itr += 1
        total_energy = 0
        for v in range(nb_views):
            indices_rot = np.random.choice(range(M_rot), p=imp_distrs_rot[v], size=params_learning_alg.N_rot)
            energies = np.zeros(params_learning_alg.N_rot)
            best_energy = 10 ** 10
            best_idx_rot = 0
            for k, idx_rot in enumerate(indices_rot):
                psi = psis[idx_rot]
                rot_vec = [true_rot_vecs[v, 0], true_rot_vecs[v, 1], psi]
                rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
                #energy = volume_representation.\
                    #get_energy_2(rot_mat, true_trans_vecs[v], views_fft[v], known_trans)
                energy, _ = volume_representation. \
                    get_energy(rot_mat, true_trans_vecs[v], views[v], recorded_shifts, v, False, known_trans)

                energies[k] = energy
                if energy < best_energy:
                    best_energy = energy
                    best_idx_rot = idx_rot
            psi = psis[best_idx_rot]
            rot_vec = [true_rot_vecs[v, 0], true_rot_vecs[v, 1], psi]
            rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
            energy = volume_representation.one_gd_step(rot_mat, true_trans_vecs[v], views, params_learning_alg.lr, known_trans, v, recorded_shifts) #,
                                                       # suppress_gauss=(v==nb_views-1))

            total_energy += energy
            energies = normalize(energies, max=6)
            likekihoods = np.exp(-energies)
            K = np.zeros((params_learning_alg.N_rot, M_rot))
            for k, idx_rot in enumerate(indices_rot):
                a = psis[idx_rot]
                K[k, :] = one_d_gaussian(psis, a, params_learning_alg.std_rot)
            update_imp_distr(imp_distrs_rot, likekihoods, K, unif_prop, M_rot, v)
        imp_distrs_rot_recorded.append(copy.deepcopy(imp_distrs_rot))
        unif_prop/=params_learning_alg.dec_factor
        if unif_prop < unif_prop_min:
            unif_prop = unif_prop_min
        total_energy /= nb_views
        print(f"total energy epoque {itr}", total_energy)
        recorded_energies.append(total_energy)
    return imp_distrs_rot_recorded, recorded_energies, recorded_shifts, unif_prop, volume_representation, itr


def gradient_descent_importance_sampling_known_rot(volume_representation, uniform_sphere_discretization, true_rot_vecs, true_trans_vecs, views,
                                                   imp_distrs, unif_prop, unif_prop_min, params_learning_alg, known_trans):
    thetas, phis, psis = uniform_sphere_discretization
    x, y, z = conversion_2_first_eulers_angles_cartesian(thetas, phis)
    axes = np.array([x, y, z])
    M = len(thetas)
    imp_distrs_recorded = []
    itr = 0
    recorded_energies = []
    nb_views = len(views)
    recorded_shifts = [[] for _ in range(nb_views)]
    while itr < params_learning_alg.N_iter_max and (not stopping_criteria(recorded_energies, params_learning_alg.eps)):
        itr += 1
        total_energy = 0
        for v in range(nb_views):
        # for v in np.random.permutation(nb_views):
            indices = np.random.choice(range(M), p=imp_distrs[v], size=params_learning_alg.N_axes)
            energies = np.zeros(params_learning_alg.N_axes)
            best_energy = 10 ** 10
            best_idx = 0
            for k, idx in enumerate(indices):
                theta, phi = thetas[idx], phis[idx]
                rot_vec = [theta, phi, true_rot_vecs[v, 2]]
                rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
                energy, _ = volume_representation. \
                    get_energy(rot_mat, true_trans_vecs[v], views[v], recorded_shifts, v, False, known_trans)

                energies[k] = energy
                if energy < best_energy:
                    best_energy = energy
                    best_idx = idx

            theta, phi = thetas[best_idx], phis[best_idx]
            rot_vec = [theta, phi, true_rot_vecs[v, 2]]
            rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
            energy = volume_representation.one_gd_step(rot_mat, true_trans_vecs[v], views, params_learning_alg.lr, known_trans, v, recorded_shifts)
            total_energy += energy
            energies = normalize(energies, max=6)
            likekihoods = np.exp(-energies)
            K_axes = \
                np.exp(params_learning_alg.coeff_kernel * axes[:, indices].T.dot(axes))

            update_imp_distr(imp_distrs, likekihoods, K_axes, unif_prop, M, v)
        imp_distrs_recorded.append(copy.deepcopy(imp_distrs))
        unif_prop /= params_learning_alg.dec_factor
        if unif_prop < unif_prop_min:
            unif_prop = unif_prop_min
        total_energy /= nb_views
        print(f'total energy epoque {itr}', total_energy)
        recorded_energies.append(total_energy)
    return imp_distrs_recorded, recorded_energies, recorded_shifts, unif_prop, volume_representation, itr


def gd_importance_sampling_3d(volume_representation, uniform_sphere_discretization, true_trans_vecs, views, imp_distrs_axes,
                              imp_distrs_rot, unif_prop,  unif_prop_min, params_learning_alg, known_trans, output_dir, ground_truth_path=None,
                              file_names=None, folder_views_selected=None, use_gpu=False):
    """volume_representation : a class representation of the volume. It must be either an objet of the class
    Fourier_pixel_representation (in file pixel_representation.py, folder volume_representation) or of the class GMM_representation
     (in file GMM_representation.py)

     uniform_sphere_discretization : this is supposed to be an almost uniform discretization of the sphere. It is produced
     via the function discretize_sphere_uniformly in file rotation_translation.py (folder common_image_processing_methods)

     true_trans_vecs : shape (Nb_views, 3). If the parameters known_trans is True, tru_trans_vecs will be used as the known values of translation
     vectors

     views : the views used for reconstruction. Shape (Nb_views, S, S, S)

     imp_distrs_axes : Shape (Nb_views, M_axes) the initial importance distributions associated to the axes of the
     3d representation of the rotation. It can be initialized as uniformed distribution

     imp_distrs_rot : Shape (Nb_views, M_rot). The initial importance distribution associated to the rotation of the
     3d representation of the rotation. It can be initialized as uniformed sitribution.

     unif_prop : initial proportion of the uniform distribution. Recommanded to initialize it to 1

     unif_prop_min : minimum values that unif_prop can reach

     params_learning_alg : object that regroups all the hyperparameters used by the function. This object is produced from
     teh class ParametersMainAlg in file class_with_parameter.py

     output_dir : results will be saved here.
     """

    if folder_views_selected is None:
        folder_views_selected = f'{output_dir}/views_selected'
        make_dir(folder_views_selected)

    make_dir(output_dir)
    print('number of views', len(views))
    thetas, phis, psis = uniform_sphere_discretization
    x, y, z = conversion_2_first_eulers_angles_cartesian(thetas, phis)
    axes = np.array([x, y, z])
    M_axes = len(thetas)
    M_rot = len(psis)
    imp_distrs_axes_recorded = []
    imp_distrs_rot_recorded = []
    recorded_energies = []
    energies_each_view = [[] for _ in range(len(views))]
    itr = 0

    recorded_shifts = [[] for _ in range(len(views))]
    ssims = []
    sub_dir = f'{output_dir}/intermediar_results'
    make_dir(sub_dir)
    ests_rot_vecs = []
    nb_step_of_supress = 0
    t0 = time.time()
    while itr < params_learning_alg.N_iter_max and (not stopping_criteria(recorded_energies, params_learning_alg.eps)):
        print(f'nb views epoch {itr} : ', len(views))

        nb_views = len(views)
        output_name = f'recons_epoch_{itr}'
        if itr%5 == 0:
            volume_representation.register_and_save(sub_dir, output_name, ground_truth_path=ground_truth_path)
        itr += 1
        total_energy = 0
        estimated_rot_vecs_iter = np.zeros((nb_views, 3))
        for v in range(nb_views):

            indices_axes = np.random.choice(range(M_axes),
                                                p=imp_distrs_axes[v], size=params_learning_alg.N_axes)
            indices_rot = np.random.choice(range(M_rot), p=imp_distrs_rot[v], size=params_learning_alg.N_rot)

            energies = np.zeros((params_learning_alg.N_axes, params_learning_alg.N_rot))
            best_energy = 10 ** 10
            best_idx_axes = 0
            best_idx_rot = 0
            true_trans_vec = true_trans_vecs[v]
            for j, idx_axes in enumerate(indices_axes):
                for k, idx_rot in enumerate(indices_rot):
                    rot_vec = [thetas[idx_axes], phis[idx_axes], psis[idx_rot]]
                    rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
                    if use_gpu:
                        view_gpu = cp.array(views[v])
                        rot_mat_gpu = cp.array(rot_mat)
                        true_trans_vec = cp.array(true_trans_vec)
                        energy_gpu, _ = volume_representation. \
                            get_energy_gpu(rot_mat_gpu, true_trans_vec, view_gpu, None, None, False, known_trans, interp_order=params_learning_alg.interp_order)
                        energy = energy_gpu.get()
                    else:
                        energy, _ = volume_representation. \
                            get_energy(rot_mat, true_trans_vec, views[v], recorded_shifts, v, False, known_trans, interp_order=params_learning_alg.interp_order)
                    energies[j, k] = energy
                    if energy < best_energy:
                        best_energy = energy
                        best_idx_axes = idx_axes
                        best_idx_rot = idx_rot

            energies = normalize(energies, max=6)
            likelihoods = np.exp(-energies)
            rot_vec = [thetas[best_idx_axes], phis[best_idx_axes], psis[best_idx_rot]]

            estimated_rot_vecs_iter[v, :] = rot_vec
            rot_mat = get_rotation_matrix(rot_vec, params_learning_alg.convention)
            energy = volume_representation.one_gd_step(rot_mat, true_trans_vec, views, params_learning_alg.lr, known_trans, v, recorded_shifts, interp_order=params_learning_alg.interp_order)
            total_energy += energy
            energies_each_view[v].append(energy)
            phi_axes = likelihoods.dot(1 / imp_distrs_rot[v][indices_rot]) / params_learning_alg.N_rot
            phi_rot = likelihoods.T.dot(1 / imp_distrs_axes[v][indices_axes]) / params_learning_alg.N_axes
            K_axes = np.exp(params_learning_alg.coeff_kernel_axes * axes[:, indices_axes].T.dot(axes))
            K_rot = np.zeros((params_learning_alg.N_rot, M_rot))

            for k, idx_rot in enumerate(indices_rot):
                a = psis[idx_rot]
                if params_learning_alg.gaussian_kernel:
                    K_rot[k, :] = one_d_gaussian(psis, a, params_learning_alg.coeff_kernel_rot)
                else:
                    K_rot[k, :] = np.exp(np.cos(a-psis)*params_learning_alg.coeff_kernel_rot)

            update_imp_distr(imp_distrs_axes, phi_axes, K_axes,unif_prop, M_axes, v)
            update_imp_distr(imp_distrs_rot, phi_rot, K_rot, unif_prop, M_rot, v)
            ests_rot_vecs.append(estimated_rot_vecs_iter)
        t = time.time()
        print('temps iter', t-t0)
        t0 = t

        if params_learning_alg.epochs_of_suppression is not None and len(params_learning_alg.epochs_of_suppression) > 0 and itr == params_learning_alg.epochs_of_suppression[0]:
            nb_step_of_supress+=1
            prop_to_suppress = params_learning_alg.proportion_of_views_suppressed.pop(0)
            nb_views_to_suppress = int(len(views)*prop_to_suppress)
            params_learning_alg.epochs_of_suppression.pop(0)
            energies_each_views_current_iter = np.array(energies_each_view)[:, -1]
            print('energies each views', energies_each_views_current_iter)
            idx_views_to_keep = np.argsort(energies_each_views_current_iter)[:len(energies_each_views_current_iter)-nb_views_to_suppress]
            print('idx kepts', idx_views_to_keep)
            views = [views[idx] for idx in idx_views_to_keep]
            imp_distrs_axes = [imp_distrs_axes[idx] for idx in idx_views_to_keep]
            imp_distrs_rot = [imp_distrs_rot[idx] for idx in idx_views_to_keep]
            energies_each_view = [energies_each_view[idx] for idx in idx_views_to_keep]
            recorded_shifts = [recorded_shifts[idx] for idx in idx_views_to_keep]
            file_names = [file_names[idx] for idx in idx_views_to_keep]
            folder_views_selected_step = f'{folder_views_selected}/step_{nb_step_of_supress}'
            make_dir(folder_views_selected_step)
            for i, fn in enumerate(file_names):
                save(f'{folder_views_selected_step}/{fn}', views[i])



        write_array_csv(estimated_rot_vecs_iter, f'{sub_dir}/estimated_rot_vecs_epoch_{itr}.csv')
        if ground_truth_path is not None:
            regist_im = io.imread(f'{sub_dir}/{output_name}_registered.tif')
            gt = io.imread(ground_truth_path)
            ssim_gt_recons = ssim(normalize(gt), normalize(regist_im))
            ssims.append(ssim_gt_recons)
        imp_distrs_rot_recorded.append(copy.deepcopy(imp_distrs_rot))
        imp_distrs_axes_recorded.append(copy.deepcopy(imp_distrs_axes))
        unif_prop /= params_learning_alg.dec_prop
        if params_learning_alg.N_iter_with_unif_distr is not None:
            if itr > params_learning_alg.N_iter_with_unif_distr:
                unif_prop = 0
        if unif_prop < unif_prop_min:
            unif_prop = unif_prop_min
        total_energy /= nb_views
        recorded_energies.append(total_energy)
        print(f'total energy ep {itr} : ', total_energy)
    write_array_csv(np.array(ssims), f'{output_dir}/ssims.csv')
    return imp_distrs_rot_recorded, imp_distrs_axes_recorded, recorded_energies, recorded_shifts, unif_prop, volume_representation, \
           itr, energies_each_view, views, file_names, ests_rot_vecs


def update_imp_distr(imp_distr, phi, K, prop, M, v):
    # phi = phi ** (1 / temp)
    q_first_comp = phi @ K
    q_first_comp /= np.sum(q_first_comp)
    imp_distr[v] = (1 - prop) * q_first_comp + prop * np.ones(M) / M
    return q_first_comp


