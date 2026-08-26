import numpy as np
from manage_files.read_save_files import make_dir
from mains.synthetic_data.main_pixel_representation_synthetic_data import main_pixel_representation
from manage_files.read_save_files import write_array_csv, read_csv
import matplotlib.pyplot as plt
from mains.real_data.main_pixel_representation_real_data import main_pixel_representation_real_data


def tsts_hyperparameter_real_data(folder_views, path_psf, root_save_folder, hyper_param_name, values_to_test, nb_tests_per_points,
                                  **args):
    for j,val in enumerate(values_to_test):
        for i in range(nb_tests_per_points):
            fold_results = f'{root_save_folder}/{hyper_param_name}_{val}/test_{i}'
            make_dir(fold_results)
            args = {hyper_param_name: val, **args}
            main_pixel_representation_real_data(folder_views, path_psf, fold_results, **args)


def tst_hyper_parameter(fold_ground_truth, ground_truth_name, save_folder, hyper_param_name,
                        values_to_test, nb_tests_per_points, **args):
    """used to test different values of an hyperparameter of the function  main_ctf_gmm.
    for each hyperparameter values it records the mean error to true angles and true translations and the ssim.

    hyper_param_name : string, name of the hyperparameter
    values_to_test : list of values taken by the hyperparameter to test
    nb_tests_per_points : number of tests to do for each parameter value
    save folder : statistics will be saved here
    **args : fixed hyperparameters
    ex of use :
    tst_hyper_parameter(fold_ground_truth, ground_truth_name, save_folder, 'N_rot',
                        [1,5,10,20,30,50], 25, known_trans=True, known_axes=True, lr=10**-4)
    """
    errors_angle = np.zeros((len(values_to_test), nb_tests_per_points))
    # errors_trans = np.zeros((len(values_to_test), nb_tests_per_points))
    # errors_2_first_euler_angles = np.zeros((len(values_to_test), nb_tests_per_points))
    make_dir(save_folder)
    ssims = np.zeros((len(values_to_test), nb_tests_per_points))
    folder_results_root = f"../../results/test_parameter_{hyper_param_name}_{ground_truth_name}"
    folder_views = f'{folder_results_root}/views'
    for j, val in enumerate(values_to_test):
        for i in range(nb_tests_per_points):
            args = {hyper_param_name: val, **args}
            folder_results = f'{folder_results_root}/val_{values_to_test[j]}_{i}'
            ssim_to_gt, mean_errors_angles = main_pixel_representation(fold_ground_truth, ground_truth_name, folder_results, folder_views,
                                                                                                                     **args)
            errors_angle[j, i] = mean_errors_angles[-1]
            ssims[j, i] = ssim_to_gt

    errors_angle_csv_path = f'{save_folder}/errors_angle.csv'
    values_to_test_csv_path = f'{save_folder}/{hyper_param_name}.csv'
    write_array_csv(values_to_test, values_to_test_csv_path)
    write_array_csv(errors_angle, errors_angle_csv_path)
    ssims_csv_path = f'{save_folder}/ssims.csv'
    write_array_csv(ssims, ssims_csv_path)
    if len(values_to_test) >= 2:
        plot_mean_error_graph_with_csv(save_folder, errors_angle_csv_path, values_to_test_csv_path,
                                       hyper_param_name)


def plot_mean_error_graph_with_csv(save_folder, errors_csv_path, values_to_test_csv_path, hyper_param_name, var_measured='angles'):
    errors = read_csv(errors_csv_path)
    values_to_test = read_csv(values_to_test_csv_path)
    mean_error = np.mean(errors, axis=1)
    std_error = np.std(errors, axis=1)
    delta = 1.96 * std_error / np.sqrt(errors.shape[1])
    conf_intervall_up = mean_error + delta
    conf_intervall_down = mean_error - delta
    plt.fill_between(values_to_test, conf_intervall_down, conf_intervall_up, alpha=0.2)
    plt.plot(values_to_test, mean_error, marker='*')
    plt.xlabel(hyper_param_name)
    unity = '(pixels)' if var_measured == 'translation' else '(°)'
    plt.ylabel(f'erreur moyenne {unity}')
    plt.grid()
    plt.title('')
    if save_folder is not None:
        plt.savefig(f'{save_folder}/error_{var_measured}.png')
        plt.close()
    else:
        plt.show()

"""
fold_gt = "../../ground_truths"
tst_hyper_parameter(fold_gt, "recepteurs_AMPA.tif", f"../../results/tst_hyper_parameter/tst_lr",
                    "dec_prop",
                    [1.1,1.13,1.15], 1, known_axes=False, known_rot=False, known_trans=True, N_axes=21)
"""
"""
for N_axes in [2,5,10,15,20]:
    tst_hyper_parameter(fold_gt, "recepteurs_AMPA.tif", f"../../results/tst_hyper_parameter/test_nb_rot_N_axes_{N_axes}", "N_rot",
                        [2,5,7,10,15,20,25], 3, known_axes=False, known_rot=False, known_trans=True, N_axes=N_axes)
"""
if __name__ == '__main__':
    from manage_files.paths import *
    channel = "c1"
    fd = "c2_hand_cropped_preprocessed" if channel == "c2" else channel
    data_folder = f"/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine_raw_prep/{fd}"
    # data_folder = f'{data_folder_root}/{channel}_6_views'
    path_psf = f'{PATH_REAL_DATA}/Data_marine_raw/PSF/PSF_6_{channel}.tif'
    hyp_param_name = "coeff_kernel_axes"
    folder_results = f"/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine_raw_prep/results/test_hyp_{hyp_param_name}"
    values_to_test = [52,102]
    tsts_hyperparameter_real_data(data_folder, path_psf, folder_results, hyp_param_name, values_to_test,1, N_iter_max=13)
