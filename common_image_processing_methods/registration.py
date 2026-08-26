import time

import SimpleITK as sitk
from numpy import pi
import os
from skimage import io
import numpy as np
from scipy.ndimage.fourier import fourier_shift
import cc3d
from manage_files.read_save_files import read_image, save
from common_image_processing_methods.rotation_translation import rotation, get_3d_rotation_matrix
from manage_files.read_save_files import save_4d_for_chimera
from common_image_processing_methods.barycenter import center_barycenter
from skimage.registration import phase_cross_correlation
from manage_files.paths import *

def registration_exhaustive_search(fixed_image_arr, moving_image_arr, output_dir, output_name, nb_dim,
                                   sample_per_axis=40, gradient_descent=False, save_res=True):
    #fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32, imageIO="TIFFImageIO")
    #moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32, imageIO="TIFFImageIO")
    fixed_image = sitk.GetImageFromArray(fixed_image_arr)
    moving_image = sitk.GetImageFromArray(moving_image_arr)
    trans = sitk.Euler3DTransform() if nb_dim == 3 else sitk.Euler2DTransform()
    trans.SetComputeZYX(True)
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          trans,
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)


    R = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)

    R.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    if not gradient_descent:
        if nb_dim == 2:
            R.SetOptimizerAsExhaustive([sample_per_axis, sample_per_axis, 0, 0])
            R.SetOptimizerScales(
                [2.0 * pi / sample_per_axis, 2.0 * pi / sample_per_axis, 1.0, 1.0])
        else:
            if isinstance(sample_per_axis, int):
                samples = [sample_per_axis, sample_per_axis, sample_per_axis, 0,0,0]
                R.SetOptimizerAsExhaustive(samples)
                R.SetOptimizerScales(
                    [2.0 * pi / sample_per_axis, 2.0 * pi / sample_per_axis,
                     2.0 * pi / sample_per_axis, 1.0, 1.0, 1.0])
            else:
                samples = sample_per_axis
                R.SetOptimizerAsExhaustive(samples)
                R.SetOptimizerScales(
                    [2.0 * pi / sample_per_axis[0], 1.0,1.0, 1.0, 1.0, 1.0])

    if gradient_descent:
        R.SetOptimizerAsGradientDescent(0.1, 100)
    R.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.

    final_transform = R.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())
    registered_array = sitk.GetArrayFromImage(moving_resampled)
    #sitk.WriteImage(moving_resampled, os.path.join(output_dir, f'{output_name}.tif'))
    angle_X, angle_Y, angle_Z, t_x, t_y, t_z = final_transform.GetParameters()
    if save_res:
        save(f'{output_dir}/{output_name}.tif', registered_array)
    return np.array([angle_X, angle_Y, angle_Z]), registered_array


def shift_registration_exhaustive_search(im1, im2, t_min=-20, t_max=20, t_step=4, fourier_space=False):
    if fourier_space:
        ft1 = im1
        ft2 = im2
    else:
        ft1 = np.fft.fftn(im1)
        ft2 = np.fft.fftn(im2)
    trans_vecs = np.arange(t_min, t_max, t_step)
    grid_trans_vec = np.array(np.meshgrid(trans_vecs, trans_vecs, trans_vecs)).T.reshape((len(trans_vecs)**3, 3))
    print('len', len(grid_trans_vec))
    min_err = 10**20
    best_i = 0
    for i,trans_vec in enumerate(grid_trans_vec):
        ft2_shifted = fourier_shift(ft2, trans_vec)
        err = np.linalg.norm(ft2_shifted-ft1)
        if err < min_err:
            best_i = i
            min_err = err
    res = fourier_shift(ft2, grid_trans_vec[best_i])
    if not fourier_space:
        res = np.fft.ifftn(res)
    return grid_trans_vec[best_i], res


def translate_to_have_one_connected_component(im, t_min=-20, t_max=20, t_step=4):
    ft = np.fft.fftn(im)
    trans_vecs = np.arange(t_min, t_max, t_step)
    grid_trans_vec = np.array(np.meshgrid(trans_vecs, trans_vecs, trans_vecs)).T.reshape((len(trans_vecs) ** 3, 3))
    number_connected_components = np.zeros(len(grid_trans_vec))
    print('len', len(number_connected_components))
    for i, trans_vec in enumerate(grid_trans_vec):
        # print('i', i)
        ft_shifted = fourier_shift(ft, trans_vec)
        im_shifted = np.fft.ifftn(ft_shifted)
        im_shifted_thresholded = 1*(im_shifted>0.2)
        _, N = cc3d.connected_components(im_shifted_thresholded, return_N=True)
        number_connected_components[i] = N

    indicices_one_component = np.where(number_connected_components==1)
    if len(indicices_one_component) == 1:
        indicices_one_component = np.where(number_connected_components==np.min(number_connected_components))
    print('idx', indicices_one_component)
    transvecs_one_components = grid_trans_vec[indicices_one_component]
    avg_transvec_one_component = np.mean(transvecs_one_components, axis=0)
    ft_shifted = fourier_shift(ft, avg_transvec_one_component)
    return np.fft.ifftn(ft_shifted)


def rotation_4d(to_rotated, rot_mat):
    rot_vol = []
    for t in range(len(to_rotated)):
        rotated, _ = rotation(to_rotated[t], rot_mat)
        rot_vol.append(rotated)
    rot_vol = np.array(rot_vol)
    return rot_vol


def register_recons_4d(est_vol, true_vol, save_fold, save_name, rot_mat=None, shift=None, rot_mat2=None):
    """est vol : 4d array shape (nb views, S, S, S)  (heterogene reconstruction)
    true vol : ground truth object
    registere est vol on true vol"""
    id = est_vol.shape[0]//2
    est_vol_middle = est_vol[id, :,:,:]
    true_vol_middle = true_vol[id, :,:,:]
    if rot_mat is None:
        print('first registration')
        rot_vec, reg_array = registration_exhaustive_search(true_vol_middle, est_vol_middle, '', '', 3, save_res=False, sample_per_axis=40)

        """
        print('shifting')
        shift, _, _ = phase_cross_correlation(true_vol_middle, reg_array)
        shifted_im = np.fft.ifftn(fourier_shift(np.fft.fftn(reg_array), shift)).real
        
        
        shifted_im = shifted_im.astype('float32')
        """
        print('second registration')
        """"""
        rot_vec2, reg_array2 = registration_exhaustive_search(true_vol_middle, reg_array, '', '', 3, save_res=False,
                                                    gradient_descent=True)
        print('shift estim')
        shift, _ = shift_registration_exhaustive_search(true_vol_middle, reg_array2, t_min=-5, t_max=5, t_step=1)
        #shift, _, _ = phase_cross_correlation(true_vol_middle, est_vol_middle, upsample_factor=10)
        rot_mat = get_3d_rotation_matrix(np.degrees(rot_vec), convention='ZYX')
        rot_mat2 = get_3d_rotation_matrix(np.degrees(rot_vec2), convention='ZYX')
    print('rotation 4d')
    save_4d_for_chimera(est_vol, f'{PATH_PROJECT_FOLDER}/est_vol.tiff')
    print('rot mat', rot_mat)
    vol1 = rotation_4d(est_vol, rot_mat)
    save_4d_for_chimera(vol1, f'{PATH_PROJECT_FOLDER}/vol1.tiff')
    vol2 = rotation_4d(vol1, rot_mat2)
    save_4d_for_chimera(vol2, f'{PATH_PROJECT_FOLDER}/vol2.tiff')
    print('shift val', shift)
    registered_est_vol = translation4d(vol2, shift)
    if save_fold is not None:
        save_4d_for_chimera(registered_est_vol, f'{save_fold}/{save_name}_reg.tiff')

    return registered_est_vol, rot_mat, shift, rot_mat2


def translation4d(to_translate, trans):
    trans_vol = []
    print('to translate shape', to_translate.shape)
    #save_4d_for_chimera(to_translate, f'{PATH_PROJECT_FOLDER}/in_4d.tiff')
    for t in range(len(to_translate)):
        #save(f'{PATH_PROJECT_FOLDER}/imin.tiff', to_translate[t])
        ft = np.fft.fftn(to_translate[t])
        ft_shifted = fourier_shift(ft, trans)
        im_shifted = np.fft.ifftn(ft_shifted)
        #save(f'{PATH_PROJECT_FOLDER}/im out.tiff', im_shifted)
        trans_vol.append(im_shifted)

    trans_vol = np.array(trans_vol)
    #save_4d_for_chimera(trans_vol, f'{PATH_PROJECT_FOLDER}/trans_vol.tiff')
    return trans_vol


def shift_registration(moving_im, fixed_im, fourier_space=False):
    shift, _, _ = phase_cross_correlation(fixed_im, moving_im)
    if fourier_space:
        ft_moving = moving_im
    else:
        ft_moving = np.fft.fftn(moving_im)
    ft_shifted = fourier_shift(ft_moving, shift)
    return np.fft.ifftn(ft_shifted)



if __name__ == '__main__':
    from common_image_processing_methods.barycenter import center_barycenter
    from skimage.registration import phase_cross_correlation

    from skimage.metrics import structural_similarity as ssim
    from manage_files.read_save_files import read_4d
    from common_image_processing_methods.others import normalize
    from skimage.restoration import denoise_tv_chambolle
    from metrics_and_visualisation.plot_conical_fsc import plot_conical_fsc

    pth = '/home/eloy/Documents/documents latex/these/images/fluofire/heterogene_recons_real_data_bicanal'
    est_vol_c1 = read_4d(f'{pth}/4d_vol_channel_1.tiff')
    est_vol_c2 = read_4d(f'{pth}/4d_vol_channel_2.tiff')
    #gt_c1 = read_4d(f'{pth}/gt_channel_1.tiff')
    #gt_c2 = read_4d(f'{pth}/gt_channel_2.tiff')
    #rot_mat = get_3d_rotation_matrix([180,0,0], convention='XYZ')
    #rotgt_c1 = rotation_4d(gt_c1, rot_mat)
    #rotgt_c2 = rotation_4d(gt_c2, rot_mat)
    est_vol_c1 = est_vol_c1[::-1, :, :, :]
    est_vol_c2 = est_vol_c2[::-1, :, :, :]
    #save_4d_for_chimera(rotgt_c1, f'{pth}/rot_gtc1.tif')
    #save_4d_for_chimera(rotgt_c2, f'{pth}/rot_gtc2.tif')
    save_4d_for_chimera(est_vol_c1, f'{pth}/4d_vol1_reversed.tif')
    save_4d_for_chimera(est_vol_c2, f'{pth}/4d_vol2_reversed.tif')
    #register_recons_4d(est_vol_c1, gt_c1, pth, 'recons_registered_c1')
    #register_recons_4d(est_vol_c2, gt_c2, pth, 'recons_registered_c2')
    1/0

    pth_root = "/home/eloy/Documents/documents latex/these/images/homogene_recons_fluo_fire"
    for gt_name in ["HIV-1-Vaccine_prep", "clathrine", "emd_0680", "recepteurs_AMPA"]:
        gt_path = f'{PTH_GT}/{gt_name}.tif'
        for nm in ["recons", "recons_pose_to_pose", "recons_sym_loss"]:
            im_registered = registration_exhaustive_search(gt_path, f'{pth_root}/{gt_name}/{nm}.tif', f'{pth_root}/{gt_name}',
                                                           f'{nm}_registered', 3)
            im_registered_gradient = registration_exhaustive_search(gt_path, f'{pth_root}/{gt_name}/{nm}_registered.tif',
                                                                    f'{pth_root}/{gt_name}', f'{nm}_registered_gradient',3, gradient_descent=True)


    1/0

    pth = f'{PATH_RESULTS_SUMMARY}/voxels_reconstruction/snr/snr_0.01'
    im_to_register_path = f'{pth}/recons_registered_trans.tif'
    gt_path = f'{PTH_GT}/recepteurs_AMPA.tif'
    gt = read_image(gt_path)

    """
    im_to_register = read_image(im_to_register_path)
    
    _, registered = shift_registration_exhaustive_search(im_to_register, gt, -5,5,1)
    save(f'{pth}/recons_registered_trans.tif', registered)
    """

    #registration_exhaustive_search(gt_path, im_to_register_path, pth, 'recons_registered', 3)
    # im_to_register = ['AMPA_5p_lambda1e-3']
    for i in range(4):
        pth2 = f'{pth}/test_{i}'
        im_to_register_path = f'{pth2}/recons.tif'
        im_to_registered = read_image(im_to_register_path)
        _, registered = shift_registration_exhaustive_search(gt, im_to_registered, -10, 10, 1)
        save(f'{pth2}/recons_registered_trans.tif', registered)
        registration_exhaustive_search(gt_path, im_to_register_path, pth2, 'recons_registered', 3)
        registration_exhaustive_search(gt_path, f'{pth2}/recons_registered.tif', pth2,
                                       'recons_registered_gradient', 3, gradient_descent=True)
    1/0



    coeffs = [1.01, 1.02, 1.03, 1.04, 1.05, 1.07, 1.1, 1.2,1.3,1.5,1.7,2]
    for c in coeffs:
        for t in range(10):
            pth_root_2 = f'{PATH_RESULTS_SUMMARY}/test_dec_prop/dec_prop_{c}/test_{t}'
            pth_im = f'{pth_root_2}/recons.tif'
            pth_gt = f'{pth_root_2}/ground_truth.tif'
            registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered', 3,
                                           sample_per_axis=40, gradient_descent=False)
            registration_exhaustive_search(pth_gt, f'{pth_root_2}/recons_registered.tif', pth_root_2, f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)
            im1 = read_image(pth_gt)
            im2 = read_image(f'{pth_root_2}/recons_registered_gradient.tif')
            plot_conical_fsc(im1, im2, pth_root_2)

    1/0


    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/voxels_reconstruction/snr'
    # im_to_register = ['AMPA_5p_lambda1e-3']
    coeffs = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 0.7, 1]

    for c in coeffs:
        for t in range(10):
            pth_root_2 = f'{pth_root}/snr_{c}/test_{t}'
            pth_im = f'{pth_root_2}/recons_registered.tif'
            pth_gt = f'{pth_root_2}/ground_truth.tif'
            registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)
            im_registered = read_image(f'{pth_root_2}/recons_registered_gradient.tif')
            im_denoised = denoise_tv_chambolle(im_registered, weight= 0.1)
            save(f'{pth_root_2}/recons_denoised.tif', im_denoised)

    1/0


    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/voxels_reconstruction/coeff_kernel_rot_3'
    # im_to_register = ['AMPA_5p_lambda1e-3']
    coeffs = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for c in coeffs:
        for t in range(10):
            pth_root_2 = f'{pth_root}/coeff_kernel_rot_{c}/test_{t}'
            pth_im = f'{pth_root_2}/recons_registered.tif'
            pth_gt = f'{pth_root_2}/ground_truth.tif'
            registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)




    1/0

    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/voxels_reconstruction/coeff_kernel_rot'
    # im_to_register = ['AMPA_5p_lambda1e-3']
    coeffs = [1,5,10,15,20,25,30,40,50,60,70,80,90,100]

    for c in coeffs:
        for t in range(10):
            pth_root_2 = f'{pth_root}/coeff_kernel_rot_{c}/test_{t}'
            pth_im = f'{pth_root_2}/recons_registered.tif'
            pth_gt = f'{pth_root_2}/ground_truth.tif'
            registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)
    1 / 0

    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/voxels_reconstruction/dec_prop'
    # im_to_register = ['AMPA_5p_lambda1e-3']
    coeffs = [1.01, 1.02, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 2]

    for c in coeffs:
        for t in range(10):
            pth_root_2 = f'{pth_root}/dec_prop_{c}/test_{t}'
            pth_im = f'{pth_root_2}/recons_registered.tif'
            pth_gt = f'{pth_root_2}/ground_truth.tif'
            registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)
    1/0

    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/cryo_ransac_reconstruction'
    for part_name in ["recepteurs_AMPA", "HIV-1-Vaccine_prep", "emd_0680", "clathrine"]:
        for test in range(11):
            pth_root_2 = f'{pth_root}/{part_name}/test_{test}'
            pth_gt = f'/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{part_name}.tif'
            pth_im = f'{pth_root_2}/recons.tif'
            registration_exhaustive_search(pth_gt, f'{pth_root_2}/recons.tif', pth_root_2,
                                           f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)

    1/0



    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/fortun_reconstructions'
    for part_name in ["recepteurs_AMPA", "HIV-1-Vaccine_prep", "emd_0680", "clathrine"]:
        pth_root_2 = f'{pth_root}/{part_name}'
        pth_gt = f'/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{part_name}.tif'
        pth_im = f'{pth_root_2}/recons.tif'
        registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered', 3,
                                       sample_per_axis=40, gradient_descent=False)
        registration_exhaustive_search(pth_gt, f'{pth_root_2}/recons_registered.tif', pth_root_2,
                                       f'recons_registered_gradient', 3,
                                       sample_per_axis=40, gradient_descent=True)


    1/0

    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/voxels_reconstruction/N_sample'
    # im_to_register = ['AMPA_5p_lambda1e-3']
    coeffs = [50,100,150,200,250,300,400,500,600,700,800,1000,1200]
    for c in coeffs:
        for t in range(10):
            pth_root_2 = f'{pth_root}/N_sample_{c}/test_{t}'
            pth_im = f'{pth_root_2}/recons_registered.tif'
            pth_gt = f'{pth_root_2}/ground_truth.tif'
            registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)
    1 / 0


    1/0

    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/voxels_reconstruction/coeff_kernel_rot_2'
    # im_to_register = ['AMPA_5p_lambda1e-3']
    coeffs = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for c in coeffs:
        for t in range(10):
            pth_root_2 = f'{pth_root}/coeff_kernel_rot_{c}/test_{t}'
            pth_im = f'{pth_root_2}/recons_registered.tif'
            pth_gt = f'{pth_root_2}/ground_truth.tif'
            registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)
    1/0

    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/voxels_reconstruction/coeff_kernel_axes_2'
    # im_to_register = ['AMPA_5p_lambda1e-3']
    coeffs = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for c in coeffs:
        for t in range(10):
            pth_root_2 = f'{pth_root}/coeff_kernel_axes_{c}/test_{t}'
            pth_im = f'{pth_root_2}/recons_registered.tif'
            pth_gt = f'{pth_root_2}/ground_truth.tif'
            registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)

    1/0
    pth_root = '/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/voxels_reconstruction/test_nb_views_anis'
    # im_to_register = ['AMPA_5p_lambda1e-3']
    nb_viewss = [2,5,10,15,20,25,30,35,40]
    sigs_z = [5,10,15,20]
    for nb_views in nb_viewss:
        for sig_z in sigs_z:
            for t in range(10):
                pth_root_2 = f'{pth_root}/nb_views_{nb_views}/sig_z_{sig_z}/test_{t}'
                pth_im = f'{pth_root_2}/recons_registered.tif'
                pth_gt = f'{pth_root_2}/ground_truth.tif'
                registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered_gradient', 3,
                                               sample_per_axis=40, gradient_descent=True)



    1/0
    gts = ["recepteurs_AMPA", "HIV-1-Vaccine_prep", "emd_0680", "clathrine"]

    for i in range(len(gts)):
        for t in range(1,30):
            pth_root_2 = f'{pth_root}/{gts[i]}/test_{t}'
            pth_im = f'{pth_root_2}/recons_registered.tif'
            pth_gt = f'{pth_root_2}/ground_truth.tif'
            registration_exhaustive_search(pth_gt, pth_im, pth_root_2, f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)
    1/0


    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/results_hpc/recepteurs_AMPA/test_nb_views_anis/test_with_pl'
    nb_viewss = [50, 60, 80]
    sigs_z = [5, 10, 15, 20]
    for nb_views in nb_viewss:
        for sig_z in sigs_z:
            pth2 = f'{pth}/nb_views_{nb_views}/sig_z_{sig_z}/test_0/recons.tif'
            im = read_image(pth2)
            translated = translate_to_have_one_connected_component(im)
            pth_save = f'{pth}/nb_views_{nb_views}/sig_z_{sig_z}/test_0/recons_translated.tif'
            save(pth_save, translated)

    1/0
    im1 = np.random.random((50,50,50))
    im2 = np.random.random((50,50,50))
    t = time.time()
    for i in range(100):
        phase_cross_correlation(im1, im2, upsample_factor=1)
    print('temps registration', time.time()-t)
    1/0

    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/results/recepteurs_AMPA/evth_unknwon/test_0'
    pth_recons = f'{pth}/recons.tif'
    part_name = "recepteurs_AMPA"
    """
    recons = read_image(pth_recons)
    
    gt = read_image(f'/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{part_name}.tif')
    _, registered = shift_registration_exhaustive_search(gt, recons)
    save(f'{pth}/recons_registered.tif', registered)
    """
    registration_exhaustive_search(f'/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{part_name}.tif', f'{pth}/recons_registered.tif',
                                   pth, 'recons_registered_trans_and_rot.tif', 3,
                                   sample_per_axis=40, gradient_descent=False)

    1/0
    pth = f'/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/results/c1_2_views/coeff_kernel_rot_50_axes_50_N_sample_512gaussian_kernel_rot'
    pth_im = f'{pth}/recons.tif'
    pth_fixed_im = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/results/c1/top_view.tif'
    output_name = 'recons_registered'
    registration_exhaustive_search(pth_fixed_im,pth_im,
                                   pth, output_name, 3,
                                   sample_per_axis=40, gradient_descent=False)


    1/0

    fold = '/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/tomographic_reconstruction'
    part_names = ["recepteurs_AMPA", "HIV-1-Vaccine_prep", "clathrine", "emd_0680"]
    for part_name in part_names:
        recons = read_image(f'{fold}/{part_name}/nb_views_10/recons.tif')
        gt_path = f'/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{part_name}.tif'
        fold_part = f'{fold}/{part_name}/nb_views_10'
        registration_exhaustive_search(gt_path, f'{fold_part}/recons.tif',
                                       fold_part, 'recons_registered', 3,
                                       sample_per_axis=40, gradient_descent=False)
        gt = read_image(gt_path)
        recons_registered = read_image(f'{fold_part}/recons_registered.tif')
        ssim_val = ssim(gt, recons_registered)
        print(f'ssim of {part_name}', ssim_val)

    """
    pth_results = f'{PATH_REAL_DATA}/Data_marine_raw_prep/c1_results/nb_views_60/test_0/set_0'
    pth1 = f'{pth_results}/intermediar_results/recons_epoch_16.tif'

    im = read_image(pth1)
    im_shifted = translate_to_have_one_connected_component(im, -40, 40, 4)
    save(f'{pth_results}/recons_shifted.tif',im_shifted)
    """
    """
    gt_name = 'recepteurs_AMPA'
    gt_path = '/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/recepteurs_AMPA.tif'
    path_init_vol = '/home/eloy/Documents/stage_reconstruction_spfluo/results/recepteurs_AMPA/sig_z_5/test_0_conv_ZXZ/recons_registered.tif'
    path_reg_vol = '/home/eloy/Documents/stage_reconstruction_spfluo/results/recepteurs_AMPA/sig_z_5/test_0_conv_ZXZ'
    
    registration_exhaustive_search(gt_path, path_init_vol, path_reg_vol,
                                   'recons_registered2', 3,
                                   sample_per_axis=50)
    
    from time import time
    from skimage import io
    from manage_files.paths import *
    from manage_files.read_save_files import *
    pth1 = f'{PATH_REAL_DATA}/picked_centrioles_preprocessed_results_0/recons.tif'
    pth2 = f'{PATH_REAL_DATA}/picked_centrioles_preprocessed_results_1/intermediar_results/recons_epoch_6.tif'
    recons_1 = read_image(pth1)
    recons_2 = read_image(pth2)
    _, registered_recons_2 = shift_registration_exhaustive_search(recons_1, recons_2, -20, 20, 4)
    print('shift registration finished')
    save(f'{PATH_REAL_DATA}/picked_centrioles_preprocessed_results_1/recons_shift_registered.tif', registered_recons_2)
    registration_exhaustive_search(pth1, pth2, f'{PATH_REAL_DATA}/picked_centrioles_preprocessed_results_1', 'recons_registered', 3)


    
    recons = io.imread(pth_recons_real_data)

    shift = np.array([0,0,7])
    # Phase-shift
    t = time()
    fourier_shifted_image = fourier_shift(np.fft.fftn(recons), shift)

    shifted_image = np.fft.ifftn(fourier_shifted_image)
    print('time shift', time() - t)
    print(shifted_image)
    pth_shifted = f'{pth_folder_res}/recons_shifted.tif'
    save(pth_shifted, shifted_image)
    """