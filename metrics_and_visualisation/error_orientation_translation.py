import matplotlib.pyplot as plt
import numpy as np

# import mains.synthetic_data.main_pixel_representation_synthetic_data
from common_image_processing_methods.rotation_translation import \
    get_rotation_matrix, get_rot_vec_from_rot_mat

from manage_files.read_save_files import write_array_csv
from scipy.interpolate import griddata
from manage_matplotlib.colors import gen_colors
from metrics_and_visualisation.fourier_shell_correlation import plot_map
from manage_matplotlib.graph_setup import *
from manage_matplotlib.graph_setup import set_up_graph

def plot_energies(energies, results_folder):
    plt.plot(energies)
    plt.xlabel('époques')
    plt.ylabel('energies')
    plt.yscale('log')
    plt.title("évolution de l'énergie au cours de l'apprentissage")
    plt.grid()
    plt.savefig(fname=f'{results_folder}/energy.png')
    plt.close()


def get_argmax_imp_distrs(imp_distrs_axes, imp_distrs_rot, view, true_rot_vecs, axes_discretization,
                          rot_discretization, itr, nb_dim):

    if imp_distrs_axes == []:
        axe_found_ref = true_rot_vecs[view, [0,1]]
    else:
        axe_found_ref = axes_discretization[np.argmax(imp_distrs_axes[itr][view])]
    if imp_distrs_rot == []:
        third_angle_ref = true_rot_vecs[view, 2]
    else:
        third_angle_ref = rot_discretization[np.argmax(imp_distrs_rot[itr][view])]
    angles_found = np.concatenate([axe_found_ref, np.array([third_angle_ref])]) if nb_dim==3 else third_angle_ref
    return angles_found


def compute_error_on_rotations(true_rot_vecs, imp_distrs_axes, imp_distrs_rot, axes_discretization, rot_discretization,
                               itr, nb_dim, nb_views, symmetry_rot_mats, convention):
    if symmetry_rot_mats is None:
        symmetry_rot_mats = np.eye(nb_dim)[None]
    errors = np.zeros((nb_views, nb_views))

    registered_angles = np.zeros((nb_views, nb_views, 3)) if nb_dim == 3 else np.zeros((nb_views, nb_views))
    true_rot_mats = [get_rotation_matrix(rot_vec, convention) for rot_vec in true_rot_vecs]
    estimated_rot_mats = []
    est_rot_vecs = []
    for v in range(nb_views):
        est_rot_vec = get_argmax_imp_distrs(imp_distrs_axes, imp_distrs_rot, v, true_rot_vecs, axes_discretization,
                                            rot_discretization, itr, nb_dim)
        est_rot_mat = get_rotation_matrix(est_rot_vec, convention)
        estimated_rot_mats.append(est_rot_mat)
        est_rot_vecs.append(est_rot_vec)
    for i in range(nb_views):
        for j in range(nb_views):
            smallest_error = 10**8
            best_k = 0
            error_for_each_rot_mat = []
            registered_rot_vec_for_each_rot_mat = []
            for k,rm in enumerate(symmetry_rot_mats):
                registered_rot_mat = true_rot_mats[i] @ rm @ true_rot_mats[j].T @ estimated_rot_mats[j]
                registered_rot_vec = get_rot_vec_from_rot_mat(registered_rot_mat, convention)
                registered_rot_vec_for_each_rot_mat.append(registered_rot_vec)
                # registered_angles[i, j] =
                est_rot_vec = est_rot_vecs[i]
                error = np.mean(np.abs(modulo_array(registered_rot_vec-est_rot_vec, 360, -180)))
                error_for_each_rot_mat.append(error)
                if error < smallest_error:
                    smallest_error = error
                    best_k = k
            error = error_for_each_rot_mat[best_k]
            errors[i, j] = error
            registered_angles[i,j] = registered_rot_vec_for_each_rot_mat[best_k]
    best_view = np.argmin(np.sum(errors), axis=0)
    mean_error = np.mean(errors[:, best_view])
    registered_angles_best_view = registered_angles[:, best_view, :]
    return mean_error, errors[:, best_view], registered_angles_best_view, est_rot_vecs


def modulo(x, mod, min=0):
    max = min + mod
    if min <= x <= max:
        return x
    elif x < min:
        x += mod
        return modulo(x, mod, min)
    elif x>max:
        x -= mod
        return modulo(x, mod, min)


def modulo_array(arr, mod, min=0):
    return [modulo(arr[i],mod, min) for i in range(len(arr))]


def plot_importance_distributions_2(fold, true_rot_vecs, imp_distrs_axes, imp_distrs_rot, nb_dim, nb_iter_ctf_steps, uniform_sphere_discretization, symmetry_rot_mats,
                                    convention, partition_rot_graphs=None):


    assert imp_distrs_rot != [] or imp_distrs_axes != []
    if imp_distrs_axes != []:
        imp_distrs_axes = np.array(imp_distrs_axes)
        # write_array_csv(imp_distrs_axes[-1], f'{fold}/imp_distrs_axes.csv')
        nb_iters, nb_views, _ = imp_distrs_axes.shape
    if imp_distrs_rot != []:
        imp_distrs_rot = np.array(imp_distrs_rot)
        write_array_csv(imp_distrs_rot[-1], f'{fold}/imp_distrs_rot.csv')
        nb_iters, nb_views, _ = imp_distrs_rot.shape
    if partition_rot_graphs is None:
        partition_rot_graphs = [range(nb_views)]
    discrete_thetas_set, discrete_phis_set, rot_discretization = uniform_sphere_discretization
    write_array_csv(true_rot_vecs, f'{fold}/true_rot_vecs.csv')
    axes_discretization = np.array([discrete_thetas_set, discrete_phis_set]).T

    mean_errors = np.zeros(nb_iters)
    for itr in range(nb_iters):
        mean_error, errors_each_view, registered_angles, est_rot_vecs = compute_error_on_rotations(true_rot_vecs, imp_distrs_axes, imp_distrs_rot, axes_discretization, rot_discretization,
                               itr, nb_dim, nb_views, symmetry_rot_mats, convention)
        mean_errors[itr] = mean_error

    write_array_csv(registered_angles, f'{fold}/registered_angles.csv')
    plot_errors(mean_errors, '°', 'three euler angles', fold, nb_iter_ctf_steps)
    if imp_distrs_axes != []:
        imp_distrs_axes_last_iter = imp_distrs_axes[-1]
        set_up_graph(MEDIUM_SIZE=40)
        plot_map(np.sum(imp_distrs_axes_last_iter, axis=0), discrete_thetas_set, discrete_phis_set, format='%.0e')
        true_thetas = modulo_array(registered_angles[:, 0],360)
        true_phis = modulo_array(registered_angles[:, 1],360)
        plt.scatter(true_thetas, true_phis, color='yellow', marker='x', s=300)
        plt.savefig(f'{fold}/map_2_first_angles_estimation.png')
        plt.close()
        #plt.show()

    print(est_rot_vecs)
    third_angle_est = np.array(est_rot_vecs)[: ,2]
    increasing_order_est_rot = np.argsort(third_angle_est)
    print('trd', third_angle_est)
    print(increasing_order_est_rot)

    """
    ax = plt.axes(projection='3d')
    imp_distrs_rot_last_iter = imp_distrs_rot[-1]
    if imp_distrs_rot != []:
        colors = gen_colors(nb_views)
        for i, v in enumerate(increasing_order_est_rot):
            Y = [i] * len(rot_discretization)
            ax.plot3D(rot_discretization, Y, imp_distrs_rot_last_iter[v, :], c=colors[i])
            if nb_dim == 3:
                true_third_angle = registered_angles[v, 2] % 360
            else:
                true_third_angle = registered_angles[v] % 360
            vertical_bar(ax, np.max(imp_distrs_rot_last_iter[v, :]), true_third_angle, i, colors[i])
        ax.view_init(30, 90)
        plt.show()
    """
    for t in range(len(partition_rot_graphs)):
        colors = gen_colors(len(partition_rot_graphs[t]))
        set_up_graph(MEDIUM_SIZE=40)
        for i,v in enumerate(partition_rot_graphs[t]):
            if nb_dim == 3:
                true_third_angle = registered_angles[v, 2]%360
            else:
                true_third_angle = registered_angles[v]%360

            plt.plot(rot_discretization, imp_distrs_rot[-1, v, :], color=colors[i], label=f'idx {v}')
            plt.vlines(true_third_angle, 0, np.max(imp_distrs_rot[-1, v, :]),
                       color=colors[i], linestyle='dashdot', linewidth=3)


        plt.grid()
        #plt.legend()
        plt.xlabel('ψ (°)')
        plt.ylabel("estimated likelihoods")
        """
        plt.rcParams['text.usetex'] = True
        plt.title(r"importance distributions $\mathcal{Q}^{l,\psi}$")
        plt.rcParams['text.usetex'] = True
        """
        plt.savefig(f'{fold}/imp_distr_angle_{t}.png')
        plt.close()

    write_array_csv(errors_each_view, f'{fold}/error_each_view.csv')
    write_array_csv(registered_angles, f'{fold}/registered_angles.csv')
    return mean_errors, errors_each_view, est_rot_vecs


def register_and_compute_error_on_rotations(true_rot_vecs, imp_distrs_axes, imp_distrs_rot,view_ref, view,
                  axes_discretization, rot_discretization, itr, nb_dim):
    angles_found_ref = get_argmax_imp_distrs(imp_distrs_axes, imp_distrs_rot, view_ref, true_rot_vecs, axes_discretization,
                          rot_discretization, itr, nb_dim)
    true_rot_mat_ref = get_rotation_matrix(true_rot_vecs[view_ref])
    estimated_rot_mat_ref = get_rotation_matrix(angles_found_ref)
    rot_mat_diff = np.linalg.inv(estimated_rot_mat_ref).dot(true_rot_mat_ref)
    angles_found_view = get_argmax_imp_distrs(imp_distrs_axes, imp_distrs_rot, view, true_rot_vecs, axes_discretization,
                          rot_discretization, itr, nb_dim)
    rot_mat_found = get_rotation_matrix(angles_found_view)
    rotated_rot_mat = rot_mat_found.dot(rot_mat_diff)

    registered_angle_found = get_rot_vec_from_rot_mat(rotated_rot_mat)
    if nb_dim == 3:
        registered_angle_found_modulo = np.array([registered_angle_found[i] % 360 for i in range(3)])
    else:
        registered_angle_found_modulo = registered_angle_found % 360
    diff = registered_angle_found_modulo - angles_found_view
    error = np.mean(np.abs(true_rot_vecs[view] - registered_angle_found_modulo))
    return error, diff, angles_found_view


def find_reference_view_minimizing_error(true_rot_vecs, imp_distrs_axes, imp_distrs_rot,
                        axes_discretization, rot_discretization, nb_views, nb_dim):
    best_view = 0
    best_end_mean_error = 500
    for v1 in range(nb_views):
        end_mean_error = 0
        for v2 in range(nb_views):
            error, _, _ = register_and_compute_error_on_rotations(true_rot_vecs, imp_distrs_axes, imp_distrs_rot,v1, v2,
                  axes_discretization, rot_discretization, -1, nb_dim)
            end_mean_error += error
        if end_mean_error <= best_end_mean_error:
            best_end_mean_error = end_mean_error
            best_view = v1
    return best_view




"""
def plot_importance_distributions(fold, true_rot_vecs, imp_distrs_axes, imp_distrs_rot, nb_dim, nb_iter_ctf_steps, uniform_sphere_discretization):
    assert imp_distrs_rot != [] or imp_distrs_axes != []
    if imp_distrs_axes != []:
        imp_distrs_axes = np.array(imp_distrs_axes)
        nb_iters, nb_views, _ = imp_distrs_axes.shape
    if imp_distrs_rot != []:
        imp_distrs_rot = np.array(imp_distrs_rot)
        nb_iters, nb_views, _ = imp_distrs_rot.shape
    discrete_thetas_set, discrete_phis_set, rot_discretization = uniform_sphere_discretization
    axes_discretization = np.array([discrete_thetas_set, discrete_phis_set]).T
    best_view = find_reference_view_minimizing_error(true_rot_vecs, imp_distrs_axes, imp_distrs_rot,
                        axes_discretization, rot_discretization, nb_views, nb_dim)
    diffs = []
    mean_errors = np.zeros(nb_iters)
    angles_found = np.zeros((nb_views, nb_dim))
    for itr in range(nb_iters):
        for v in range(nb_views):
            error, diff, angles_found_view = register_and_compute_error_on_rotations(true_rot_vecs, imp_distrs_axes, imp_distrs_rot,best_view, v,
                  axes_discretization, rot_discretization, itr, nb_dim)
            mean_errors[itr] += error
            if itr == nb_iters-1:
                diffs.append(diff)
                angles_found[v] = angles_found_view
    write_array_csv(angles_found, f'{fold}/angles_found.csv')
    diffs = np.array(diffs)
    mean_errors /= nb_views
    translated_true_angles = true_rot_vecs - diffs
    translated_true_angles = translated_true_angles % 360
    plot_errors(mean_errors, '°', 'three euler angles', fold, nb_iter_ctf_steps)
    if imp_distrs_axes != []:
        plot_map(axes_discretization, imp_distrs_axes)
        plt.scatter(translated_true_angles[:, 0], translated_true_angles[:, 1], color='red', marker='x')
        plt.savefig(f'{fold}/map_2_first_angles_estimation.png')
        plt.close()
    if imp_distrs_rot != []:
        colors = gen_colors(nb_views)
        for v in range(nb_views):
            if nb_dim == 3:
                true_third_angle = translated_true_angles[v, 2]
            else:
                true_third_angle = translated_true_angles[v]
            plt.plot(rot_discretization, imp_distrs_rot[-1, v, :], color=colors[v])
            plt.vlines(true_third_angle, 0, np.max(imp_distrs_rot[-1, v, :]),
                       color=colors[v])
        plt.grid()
        plt.xlabel('ψ (°)')
        plt.ylabel("distributions d'importance")
        plt.title("distributions d'importance associées à la rotation dans le plan")
        plt.savefig(f'{fold}/imp_distr_angle.png')
        plt.close()
    return mean_errors
"""


def plot_errors(mean_errors, unity, var_name, fold, nb_iter_ctf_steps):

    plt.plot(range(len(mean_errors)), mean_errors)
    if nb_iter_ctf_steps is not None:
        nb_steps = len(nb_iter_ctf_steps)
        colors = gen_colors(nb_steps)
        for s in range(nb_steps):
            plt.vlines(np.sum(nb_iter_ctf_steps[:s]), 0, np.max(mean_errors), color=colors[s],
                       label=f'step {s}')
    plt.title(f'mean error to true {var_name}')
    plt.xlabel('iterations')
    plt.ylabel(f'mean error (over all views ({unity}))')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(f'{fold}/mean_error_{var_name}.png')
    plt.close()
    write_array_csv(mean_errors, f'{fold}/mean_error_{var_name}.csv')

"""
def plot_map(axes_discretization, imp_distrs_axes):
    sum_imp_distr = np.sum(imp_distrs_axes[-1], axis=0)
    min_theta, max_theta, min_phi, max_phi = np.min(axes_discretization[:, 0]), np.max(axes_discretization[:, 0]), \
                                             np.min(axes_discretization[:, 1]), np.max(axes_discretization[:, 1])
    # min_theta, max_theta, min_phi, max_phi = 0,360,0,180
    grid_theta, grid_phi = np.mgrid[min_theta:max_theta:500j,
                           min_phi:max_phi:500j]
    grid_z = griddata(axes_discretization, sum_imp_distr, (grid_theta, grid_phi), method='cubic')

    plt.imshow(np.nan_to_num(grid_z.T), extent=(min_theta, max_theta, min_phi, max_phi),
               origin='lower')  # , origin='lower', extent=(0, 1, 0, 1))

    plt.rcParams['text.usetex'] = True
    plt.title(r'importance distributions $\mathcal{Q}^{l,d}$')
    plt.xlabel(r'$\phi_1$ (°)')
    plt.ylabel(r'$\phi_2$ (°)')
    plt.rcParams['text.usetex'] = False
"""


def mean_error_true_shift(folder, recorded_shifts, true_shifts, nb_iters_ctf_steps):

    recorded_shifts = np.array(recorded_shifts)
    nb_views, nb_iter, nb_dim = recorded_shifts.shape
    mean_error_true_shift = np.zeros(nb_iter)
    for itr in range(nb_iter):
        est_shift_iter = recorded_shifts[:, itr, :]
        mean_error = np.mean(np.abs(est_shift_iter-true_shifts))
        mean_error_true_shift[itr] = mean_error
    plot_errors(mean_error_true_shift, 'pixels', 'shift', folder, nb_iters_ctf_steps)
    return mean_error_true_shift


if __name__ == '__main__':
    from manage_files.read_save_files import read_csv
    from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly
    fold = '/home/eloy/Documents/stage_reconstruction_spfluo/results/recepteurs_AMPA/test_for_imp_distr/test_2'
    uniform_sphere_discretization = discretize_sphere_uniformly(360**2, 360)
    print('cc1')
    imp_distrs_axes = np.array([read_csv(f'{fold}/imp_distrs_axes.csv')])
    #imp_distrs_axes = []
    print('cc2')
    imp_distrs_rot = np.array([read_csv(f'{fold}/imp_distrs_rot.csv')])
    true_rot_vecs = read_csv(f'{fold}/true_rot_vecs.csv')
    print('cc3')
    #print('shp', imp_distrs_axes.shape)
    partition_rot_graphs = [[3,4,6,8,9,13,14,15,17,18], [0,1,2,5,7,10,11,12,16,19]]
    print(partition_rot_graphs)
    plot_importance_distributions_2(fold, true_rot_vecs, imp_distrs_axes, imp_distrs_rot, 3, None, uniform_sphere_discretization, None, partition_rot_graphs)








