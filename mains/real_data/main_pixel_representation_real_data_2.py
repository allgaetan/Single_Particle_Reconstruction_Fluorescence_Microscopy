import numpy as np
from manage_files.read_save_files import read_images_in_folder, make_dir
from volume_representation.pixel_representation import Fourier_pixel_representation
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import make_grid, nd_gaussian
from learning_algorithms.gradient_descent_importance_sampling import gd_importance_sampling_3d
from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly
import matplotlib.pyplot as plt
from manage_files.read_save_files import write_array_csv, make_dir_and_write_array, read_image, print_dictionnary_in_file
from metrics_and_visualisation.error_orientation_translation import plot_map, gen_colors, get_argmax_imp_distrs
from skimage import io
from manage_files.paths import *
from data_generation.generate_data import get_PSF_with_stds
from common_image_processing_methods.others import crop_center, resize, normalize
from common_image_processing_methods.rotation_translation import conversion_2_first_eulers_angles_cartesian
from manage_matplotlib.plot_graph import plot_graphs, save_figure
from metrics_and_visualisation.fourier_shell_correlation import plot_resolution_map
from metrics_and_visualisation.metrics_to_compare_2_images import fsc
from learning_algorithms.gradient_descent_known_angles import gradient_descent_known_rot
from manage_files.read_save_files import save
from time import time


def main_pixel_representation_real_data(folder_views, path_psf, folder_results, nb_dim=3, eps=1, lr=1, N_iter_max=13, N_axes=None, coeff_kernel_axes=50, N_rot=21, coeff_kernel_rot=5,
                              M_axes=10000, M_rot=360, dec_prop=1.2):
        if N_axes is None:
            N_axes = N_rot
        if coeff_kernel_rot is None:
            coeff_kernel_rot = coeff_kernel_axes
        views, file_names = read_images_in_folder(folder_views)
        size = views[0].shape[0]
        recons_2_sets = []
        if path_psf is None:
            PSF = get_PSF_with_stds(size, 4, 1)
        else:
            PSF = read_image(path_psf)
            PSF = crop_center(PSF, (size, size, size))
        PSF/=np.sum(PSF)
        print('PSF saved at loc', folder_results)
        make_dir(folder_results)
        save(f'{folder_results}/PSF.tif', PSF)
        f = open(f'{folder_results}/views_order.txt', 'w')
        print(file_names, file=f)
        fourier_pixel_rep = Fourier_pixel_representation(nb_dim, size, PSF)
        uniform_sphere_discretization = discretize_sphere_uniformly(M_axes, M_rot)
        imp_distrs_rot = np.ones((len(views), M_rot)) / M_rot
        imp_distrs_axes = np.ones((len(views), M_axes)) / M_axes
        params = locals()

        """
        volume_representation, recorded_energies, recorded_shifts, itr = \
                                                                gradient_descent_known_rot(fourier_pixel_rep,
                                                                                        np.zeros((1,3)), np.zeros((1,3)),
                                                                                                   views, eps, lr, N_iter_max, True)

        """
        t = time()
        imp_distrs_rot_recorded, imp_distrs_axes_recorded, recorded_energies, recorded_shifts, unif_prop, volume_representation, itr = \
                    gd_importance_sampling_3d(fourier_pixel_rep, uniform_sphere_discretization, None, views,
                                          imp_distrs_axes,
                                          imp_distrs_rot, 1, dec_prop, 0, N_axes, N_rot,
                                          coeff_kernel_axes, coeff_kernel_rot,
                                          N_iter_max, eps, lr, False, folder_results, N_iter_with_unif_ditr=10, gaussian_kernel=True)

        total_time = t - time()
        fourier_pixel_rep.register_and_save(folder_results, 'recons', translate=True, register=False)
        # fourier_pixel_rep.register_and_save(folder_results, 'recons', translate=True)
        recons_2_sets.append(fourier_pixel_rep.get_image_from_fourier_representation())
        discrete_thetas_set, discrete_phis_set, rot_discretization = uniform_sphere_discretization
        axes_discretization = np.array([discrete_thetas_set, discrete_phis_set]).T
        loc = f'{folder_results}/training_param_time.txt'
        f = open(loc, 'w')
        print_dictionnary_in_file(params, f)
        print(f'total time : {total_time}', file=f)
        print(f'total time : {total_time}', file=f)
        plot_map(axes_discretization, imp_distrs_axes_recorded)
        plt.savefig(f'{folder_results}/map_2_first_angles_estimation.png')
        plt.close()
        write_array_csv(recorded_energies, f'{folder_results}/energies.csv')
        plot_graphs(rot_discretization, np.array(imp_distrs_rot_recorded)[-1,:,:], 'ψ (°)', "distributions d'importance",
                    "distributions d'importance associées à la rotation dans le plan", sav_fold=folder_results, save_name='imp_distr_rot')

        angles_found = np.zeros((len(views), nb_dim))
        for v in range(len(views)):
            angles_found_view = get_argmax_imp_distrs(imp_distrs_axes_recorded, imp_distrs_rot_recorded, v, None, axes_discretization,
                                      rot_discretization, -1, nb_dim)
            angles_found[v] = angles_found_view
            print(angles_found_view)
        write_array_csv(angles_found, f'{folder_results}/angles_found.csv')

if __name__ == '__main__':
    channel = 'c1'
    #fd = "c2_hand_cropped_preprocessed" if channel == "c2" else channel
    data_folder = f"{PATH_REAL_DATA}/Data_marine_cropped_preprocessed/{channel}"
    # data_folder = f'{data_folder_root}/{channel}_6_views'
    path_psf = f'{PATH_REAL_DATA}/Data_marine_raw/PSF/PSF_6_{channel}.tif'
    folder_results = f'{PATH_REAL_DATA}/results/data_marine/test_2'
    main_pixel_representation_real_data(data_folder, path_psf, folder_results)