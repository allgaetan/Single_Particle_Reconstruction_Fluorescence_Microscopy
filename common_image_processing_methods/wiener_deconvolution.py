from numpy.fft import fftn, ifftshift, ifftn
import numpy as np

def abs2(x):
    """Returns the squared absolute value of its agument."""
    if np.iscomplexobj(x):
        x_re = x.real
        x_im = x.imag
        return x_re*x_re + x_im*x_im
    else:
        return x*x


def wiener_filter(y, h, alpha):
    tf_h = fftn(ifftshift(h))
    tf_y = fftn(y)
    print('max', np.max(abs2(tf_h) ))
    return ifftn(np.conj(tf_h)*tf_y/(abs2(tf_h) + alpha)).real


if __name__ == '__main__':
    from manage_files.paths import PATH_REAL_DATA
    from manage_files.read_save_files import read_image, save, read_multichannel
    from common_image_processing_methods.others import crop_center
    from data_generation.generate_data import get_PSF_with_stds
    from classes_with_parameters import ParametersDataGeneration

    pth = f'{PATH_REAL_DATA}/SAS6/picking/deconv_cropped_proto'
    im_name = '190315_U2OS 2X AA FA Sas6 488 Tub rabbit.lif - Lightning 009Series008_Lng.tif_2.tif'
    pth_im = f'{pth}/good_same_size_resized/{im_name}'
    size = 50
    for sig_z in np.arange(3,6,0.5):
        psf = ParametersDataGeneration(sig_z=sig_z, sig_xy=1, size=size).get_psf()
        im = read_multichannel(pth_im)
        im_c2 = im[1]
        # psf_true = psf_true/np.sum(psf_true)
        deconv = wiener_filter(im_c2, psf, 0.001)
        #deconv_psf_true = wiener_filter(im, psf_true, 0.001)
        save(f'{pth}/deconv_gaus_{sig_z}.tif', deconv)
        save(f'{pth}/psf_gauss_{sig_z}.tif', psf)
    1/0



    """
    from skimage import io
    from data_generation.generate_data import *
    from manage_files.read_save_files import *
    for gt_name in ["HIV-1-Vaccine_prep", "emd_0680", "Vcentriole_prep", "clathrine"]:
        fold_single_view = f"/home/eloy/Documents/stage_reconstruction_spfluo/views/{gt_name}/single_view"
        fold_results = f"/home/eloy/Documents/stage_reconstruction_spfluo/views/{gt_name}/single_view_deconv"
        gt_path = f"/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{gt_name}.tif"
        gt = io.imread(gt_path)
        for sig_z in [5]:#[1,3,5,7,10,12,15,17,20]:
            view_fold = f'{fold_single_view}/sig_z_{sig_z}'
            make_dir(view_fold)
            save_fold = f'{fold_results}/sig_z_{sig_z}'
            make_dir(save_fold)
            for t in range(100):
                #view = io.imread(f'{fold_single_view}/sig_z_{sig_z}/view_{t}.tif')
                PSF, view = convolve_noise(gt, sig_z, 1)
                save(f'{view_fold}/view_{t}.tif', view)
                h = get_PSF_with_stds(50, sig_z, 1)
                deconv = wiener_filter(view, h, 0.1)
                save(f'{save_fold}/view_{t}.tif', deconv)


    """
    """
    from manage_files.read_save_files import read_images_in_folder, save
    from skimage import io
    data_folder = "/home/eloy/Documents/stage_reconstruction_spfluo/real_data/high_res_images/preprocessed"
    path_psf =  '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/high_res_images/psf_preprocessed.tif'
    views, file_names = read_images_in_folder(data_folder)
    psf = io.imread(path_psf)
    deconv = wiener_filter(views[0], psf, 0)
    deconv_path = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/high_res_images/single_view_deconv.tif'
    save(deconv_path, deconv)
    """

