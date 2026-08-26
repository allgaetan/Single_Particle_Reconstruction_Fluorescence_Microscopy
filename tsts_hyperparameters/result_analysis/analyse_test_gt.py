from skimage import io
import numpy as np
from manage_files.read_save_files import read_csv
from metrics_and_visualisation.metrics_to_compare_2_images import fsc, f1_score
from skimage.metrics import structural_similarity as ssim
from common_image_processing_methods.others import normalize
from metrics_and_visualisation.fourier_shell_correlation import calc_fsc, fsc2res
from manage_files.paths import PATH_PROJECT_FOLDER
from manage_files.read_save_files import write_array_csv


gt_names = ["recepteurs_AMPA", "HIV-1-Vaccine_prep", "clathrine", "emd_0680"]

"""recons with knonwn angles"""
fold_root = f"{PATH_PROJECT_FOLDER}/results_summary/voxels_reconstruction"
nb_tests = 10

for k,gt_name in enumerate(gt_names):
    fscs = []
    ssims = []
    pth = f'{fold_root}/poses_connues/{gt_name}'
    for t in range(nb_tests):
        pth2 = f'{pth}/test_{t}'
        recons = io.imread(f'{pth2}/recons.tif')
        gt = io.imread(f'{pth2}/ground_truth.tif')
        ssim_val = ssim(recons, gt)
        fsc_val = fsc(recons, gt)
        ssims.append(ssim_val)
        fscs.append(fsc_val)
    print('fscs', fscs)
    write_array_csv(np.array([[np.mean(np.array(fscs))]]), f'{pth}/mean_fsc.csv')
    write_array_csv(np.array([[np.std(np.array(fscs))]]), f'{pth}/std_fsc.csv')
    write_array_csv(np.array([[np.mean(np.array(ssims))]]), f'{pth}/mean_ssims.csv')
    write_array_csv(np.array([[np.std(ssims)]]), f'{pth}/std_ssims.csv')


1/0


nb_tests = 30
nb_tests_scipion = 10
ssims = np.zeros((len(gt_names), nb_tests))
dices = np.zeros((len(gt_names), nb_tests))
dices2 = np.zeros((len(gt_names), nb_tests))
errors_angles = np.zeros((len(gt_names), nb_tests))
fscs_res = [[] for _ in range(len(gt_names))]
fscs_res_to_gt = np.zeros((len(gt_names), nb_tests))
f1_scores = np.zeros((len(gt_names), nb_tests))
f1_scores_scipion = np.zeros((len(gt_names), nb_tests_scipion))
ssims_scipion = np.zeros((len(gt_names), nb_tests_scipion))
fscs_res_scipion = [[] for _ in range(len(gt_names))]
fscs_res_to_gt_scipion = np.zeros((len(gt_names), nb_tests_scipion))
for k,gt_name in enumerate(gt_names):
    for i in range(nb_tests):
        fold = f'/home/eloy/Documents/stage_reconstruction_spfluo/results_hpc/test_gt/{gt_name}/test_{i}'
        recons = io.imread(f'{fold}/recons_registered.tif')
        recons[recons<0.05] = 0
        gt = io.imread(f'{fold}/ground_truth.tif')
        # ssim_val = read_csv(f'{fold}/ssims.csv')[-1]
        ssim_val = ssim(recons, gt)
        ssims[k, i] = ssim_val
        fscs_res_to_gt[k, i] = fsc(recons, gt, ground_truth_sizes[gt_name]/2)
        error_angle = read_csv(f'{fold}/mean_error_three euler angles.csv')[-1]
        errors_angles[k, i] = error_angle
        f1_scores[k, i] = f1_score(recons, gt)
        for j in range(i+1, nb_tests):
            fold2 = f'/home/eloy/Documents/stage_reconstruction_spfluo/results_hpc/test_gt/{gt_name}/test_{j}'
            recons2 = io.imread(f'{fold2}/recons_registered.tif')
            fsc_val = calc_fsc(recons2, recons, ground_truth_sizes[gt_name]/2)
            fsc_res = fsc2res(fsc_val)
            fscs_res[k].append(fsc_res)
        if i < nb_tests_scipion:
            path_recons_scipion = f"/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/tomographic_reconstruction/{gt_name}/nb_views_80/vol{i+1}_registered.tif"
            print('path recons scipion', path_recons_scipion)
            recons_scipion = io.imread(path_recons_scipion)
            print('ssim', ssim(gt, recons_scipion))
            ssims_scipion[k, i] = ssim(gt, recons_scipion)
            fscs_res_to_gt_scipion[k, i] = fsc(recons_scipion, gt, ground_truth_sizes[gt_name]/2)
            f1_scores_scipion[k, i] = f1_score(recons_scipion, gt)
            for j in range(i+1, nb_tests_scipion):
                path2 = f"/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/tomographic_reconstruction/{gt_name}/nb_views_80/vol{j+1}_registered.tif"
                recons_scipion_2 = io.imread(path2)
                fsc_val = calc_fsc(recons_scipion_2, recons_scipion, ground_truth_sizes[gt_name] / 2)
                fsc_res = fsc2res(fsc_val)
                fscs_res_scipion[k].append(fsc_res)




        """
        dice = dice_after_thresholding(recons, gt, thresh=0.2)
        dices[k, i] = dice
        dice2 = dice_after_thresholding(recons, gt, thresh=0.6)
        dices[k, i] = dice2
        """


fscs_res = np.array(fscs_res)
fscs_res_scipion = np.array(fscs_res_scipion)


nb_tests_single_view = 100
f1_scores_single_view = np.zeros((len(gt_names), nb_tests_single_view))
ssims_single_view = np.zeros((len(gt_names), nb_tests_single_view))
fscs_res_to_gt_single_view = np.zeros((len(gt_names), nb_tests_single_view))
for k,gt_name in enumerate(gt_names):
    gt_path = f"/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{gt_name}.tif"
    gt = io.imread(gt_path)
    for i in range(nb_tests_single_view):
        deconv_path = f"/home/eloy/Documents/stage_reconstruction_spfluo/views/{gt_name}/single_view_deconv/sig_z_5/view_{i}.tif"
        deconv = io.imread(deconv_path)
        f1_scores_single_view[k, i] = f1_score(deconv, gt)
        ssim_val = ssim(deconv, gt)
        ssims_single_view[k, i] = ssim_val
        fscs_res_to_gt_single_view[k, i] = fsc(deconv, gt, ground_truth_sizes[gt_name]/2)


round_nb = 3
for k in range(len(gt_names)):

    if gt_names[k] == 'Vcentriole_prep':
        print(f'ground truth {gt_names[k]}, mean error angles : 11.22, std errors angles : {np.round(np.std(errors_angles[k, :]), round_nb)}')
    else:
        print(f'ground truth {gt_names[k]}, mean error angles : {np.round(np.mean(errors_angles[k, :]), round_nb)}, std errors angles : {np.round(np.std(errors_angles[k, :]), round_nb)}')
    print(f'mean ssim our method : {np.round(np.mean(ssims[k, :]), round_nb)}, std ssim : {np.round(np.std(ssims[k, :]), round_nb)}')
    print(f'mean fsc our method : {np.round(np.mean(fscs_res[k, :]), round_nb)}, std ssim : {np.round(np.std(fscs_res[k, :]), round_nb)}')
    print(f'mean ssim single view deconvolution {np.round(np.mean(ssims_single_view[k, :]), round_nb)}, std ssim : {np.round(np.std(ssims_single_view[k, :]), round_nb)}')
    print(f'mean fsc to gt {np.round(np.mean(fscs_res_to_gt[k, :]), round_nb)}, std : {np.round(np.std(fscs_res_to_gt[k, :]), round_nb)}')
    print(f'mean fsc to gt single view {np.round(np.mean(fscs_res_to_gt_single_view[k, :]), round_nb)}, std : {np.round(np.std(fscs_res_to_gt_single_view[k, :]), round_nb)}')
    print(
        f'mean f1 score to gt single view {np.round(np.mean(f1_scores_single_view[k, :]), round_nb)}, std : {np.round(np.std(f1_scores_single_view[k, :]), round_nb)}')
    print(
        f'mean f1 score to gt  {np.round(np.mean(f1_scores[k, :]), round_nb)}, std : {np.round(np.std(f1_scores[k, :]), round_nb)}')

    print(f'mean ssim scipion : {np.round(np.mean(ssims_scipion[k, :]), round_nb)}, std  : {np.round(np.std(ssims[k, :]), round_nb)}')
    print(f'mean fsc scipion : {np.round(np.mean(fscs_res_scipion[k, :]), round_nb)}, std  : {np.round(np.std(fscs_res_scipion[k, :]), round_nb)}')
    print(f'mean f1 score to gt scipion  {np.round(np.mean(f1_scores_scipion[k, :]), round_nb)}, std : {np.round(np.std(f1_scores_scipion[k, :]), round_nb)}')
    print(f'mean fsc to gt scipion  {np.round(np.mean(fscs_res_to_gt_scipion[k, :]), round_nb)}, std : {np.round(np.std(fscs_res_to_gt_scipion[k, :]), round_nb)}')

    print(
        f'mean ssim known angles : {np.round(np.mean(ssims_known_angles[k, :]), round_nb)}, std  : {np.round(np.std(ssims_known_angles[k, :]), round_nb)}')
    print(
        f'mean fsc known_angles : {np.round(np.mean(fscs_res_known_angles[k, :]), round_nb)}, std  : {np.round(np.std(fscs_res_known_angles[k, :]), round_nb)}')
    print(
        f'mean f1 score known angles {np.round(np.mean(f1_scores_known_angles[k, :]), round_nb)}, std : {np.round(np.std(f1_scores_known_angles[k, :]), round_nb)}')


    #print(f'mean dice thresh 0.2: {np.round(np.mean(dices[k, :]), round_nb)}, std dice : {np.round(np.std(dices[k, :]), round_nb)}')
    #print(f'mean dice thresh 0.6: {np.round(np.mean(dices2[k, :]), round_nb)}, std dice : {np.round(np.std(dices2[k, :]), round_nb)}')