from numpy.fft import fftn, ifftshift, ifftn, fftshift
import numpy as np
from data_generation.generate_data import convolve_fourier
import mains.synthetic_data.main_pixel_representation_synthetic_data


def deconv_gradient_desent(blurred, psf, size, lr, nb_dim, N_iter, reg_coeff=0.01):
    psf/=np.sum(psf)
    volume_fourier = np.random.random(tuple([size for _ in range(nb_dim)]))
    volume_fourier = volume_fourier.astype(dtype=np.complex_)
    psf_fourier = np.fft.fftn(ifftshift(psf))
    blurred_fourier = np.fft.fftn(blurred)
    for i in range(N_iter):
        energy = np.linalg.norm(psf_fourier * volume_fourier - blurred_fourier) ** 2 / (size ** nb_dim)
        grad =  psf_fourier*(psf_fourier*volume_fourier-blurred_fourier)
        gradient_l2_reg = 2 * volume_fourier
        l2_reg = np.mean(np.abs(volume_fourier) ** 2)
        volume_fourier = volume_fourier - lr*grad
        volume_fourier = volume_fourier - reg_coeff*lr*gradient_l2_reg
        print(f'energy itr {i}', energy+reg_coeff*l2_reg)
    ifft =  ifftn(volume_fourier)
    return np.abs(ifft)



if __name__ == '__main__':
    from manage_files.read_save_files import read_image, save
    from data_generation.generate_data import get_PSF_with_stds
    from common_image_processing_methods.others import crop_center
    from manage_files.paths import *

    size = 50
    pth_burred = '211214_siCT_CEP164.lif - Series013.tif'
    pth_psf = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/PSFs_preprocessed/PSF_6_c2_resized_ratio_2.tif'
    psf = read_image(pth_psf)
    psf_gaussian = get_PSF_with_stds(size, 4, 1)
    save('PSF_gaussian.tif', psf_gaussian)
    blurred = read_image(pth_burred)
    psf = crop_center(psf, (50,50,50))
    save('real_psf_cropped.tif', psf)
    lr = 0.05
    """données réelles"""
    #deconv avec psf 'réelle' de Denis
    deconv = deconv_gradient_desent(blurred, psf, 50, 1.5, 3, 800)
    save('deconv_real_psf.tif', deconv)
    1/0
    # deconv avec psf gaussienne
    deconv = deconv_gradient_desent(blurred, psf_gaussian, 50, lr, 3, 200)
    save('deconv_gaussian_psf.tif', deconv)

    """données synthétiques"""
    pth_gt = f'/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/recepteurs_AMPA.tif'
    ground_truth = read_image(pth_gt)

    # deconv avec psf relle
    blurred_gt_real_psf = convolve_fourier(ground_truth, psf)
    deconv = deconv_gradient_desent(blurred_gt_real_psf, psf, 50, lr, 3, 200)
    save('deconv_synthetic_data_real_psf.tif', deconv)
    save('blurred_gt_real_psf.tif', blurred_gt_real_psf)

    # deconv avec psf gaussienne
    blurred_gt_gaussian_psf = convolve_fourier(ground_truth, psf_gaussian)
    deconv = deconv_gradient_desent(blurred_gt_gaussian_psf, psf_gaussian, 50, lr, 3, 200)
    save('deconv_synthetic_data_gaussian_psf.tif', deconv)
    save('blurred_gt_gaussian_psf.tif', blurred_gt_real_psf)

    # deconv avec PSF somme de 2 PSF gaussiennes orthogonales
    from common_image_processing_methods.rotation_translation import get_rotation_matrix, rotation

    rot_mat = get_rotation_matrix([90, 0, 0])
    orthogonal_gaussian_psf , _ = rotation(psf_gaussian, rot_mat)
    two_gauss_psf = psf + orthogonal_gaussian_psf
    blurred_gt_two_gaussians = convolve_fourier(ground_truth, two_gauss_psf)
    deconv = deconv_gradient_desent(blurred_gt_gaussian_psf, two_gauss_psf, 50, lr, 3, 200)
    save('psf_two_gaussians.tif', two_gauss_psf)
    save('deconv_synthetic_data_two_gaussians.tif', deconv)
    save('blurred_synthetic_data_2_gaussians_psf.tif', blurred_gt_two_gaussians)
