import cv2
from skimage.restoration import denoise_tv_chambolle
from manage_files.paths import PTH_LOCAL_RESULTS
from manage_files.read_save_files import save, write_array_csv, read_image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from metrics_and_visualisation.metrics_to_compare_2_images import fsc
from common_image_processing_methods.others import normalize

if __name__ == '__main__':
    path_root = f'{PTH_LOCAL_RESULTS}/test_noise'
    snr = 100
    for i in range(15):
        fold = f'{path_root}/test_{i}'
        im_to_denoise = read_image(f'{fold}/recons.tif')
        denoised_im = denoise_tv_chambolle(im_to_denoise, weight=0.5)

        gt = read_image(f'{fold}/ground_truth.tif')
        power_gt = np.mean(gt ** 2)
        std_noise = np.sqrt(power_gt / snr)
        noise = np.random.normal(0, std_noise, gt.shape)
        gt_noised = gt  + noise
        gt_noised_denoised = denoise_tv_chambolle(gt_noised, weight=0.5)
        save(f'{fold}/gt_noised_denoised.tif', gt_noised_denoised)
        print('ssikv', ssim(gt_noised, gt))
        ssim_gt_noise = ssim(gt_noised, gt)

        #save(f'{fold}/recons_denoised.tif', denoised_im_2)
        #ssim_denoised = ssim(denoised_im_2, gt)
        print('ssim denoised recons', ssim_denoised)
        fsc_denoised = fsc(denoised_im, gt)
        write_array_csv(np.array([[ssim_denoised]]), f'{fold}/ssim_denoised.csv')
        write_array_csv(np.array([[fsc_denoised]]), f'{fold}/fsc_denoised.csv')
