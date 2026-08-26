import numpy as np
try:
    import cupy as cp
except:
    import numpy as cp
from numpy import fft
from common_image_processing_methods.rotation_translation import rotation, translation, rotation_gpu
from manage_files.read_save_files import save, read_image
from common_image_processing_methods.registration import registration_exhaustive_search, shift_registration_exhaustive_search
from skimage.registration import phase_cross_correlation
try:
    from cucim.skimage.registration import phase_cross_correlation as phase_cross_correlation_gpu
except:
    from skimage.registration import phase_cross_correlation as phase_cross_correlation_gpu
from scipy.ndimage.fourier import fourier_shift
try:
    from cupyx.scipy.ndimage import fourier_shift as fourier_shift_gpu
except:
    from scipy.ndimage import fourier_shift as fourier_shift_gpu
from common_image_processing_methods.others import normalize, crop_center
from common_image_processing_methods.registration import translate_to_have_one_connected_component
from data_generation.generate_data import get_PSF_with_stds
from common_image_processing_methods.barycenter import center_barycenter
from time import time


class Fourier_pixel_representation:
    def __init__(self, nb_dim, size, psf, init_vol=None, random_init=True):
        if init_vol is None:
            if random_init:
                volume_fourier = np.random.random(tuple([size for _ in range(nb_dim)]))
            else:
                volume_fourier = get_PSF_with_stds(size, size/10, size/10)
        else:
            volume_fourier = np.fft.fftn(np.fft.ifftshift(crop_center(init_vol, (size, size, size))))
        self.volume_fourier = volume_fourier.astype(dtype=np.complex_)
        self.volume_fourier_gpu = cp.array(self.volume_fourier)
        self.nb_dim = nb_dim
        self.size = size
        self.psf = psf
        self.psf_gpu = cp.array(self.psf)
        self.psf_fft = fft.fftn(self.psf)

    def get_energy(self, rot_mat, trans_vec, view, recorded_shifts, view_idx, save_shift=False, known_trans=True, interp_order=3):
        #size_crop = 10
        #rotated_PSF = crop_center(rotation(crop_center(self.p00sf, (self.crop_size_psf, self.crop_size_psf, self.crop_size_psf)), rot_mat)[0], self.psf.shape)
        psf_rotated_fft = fft.fftn(rotation(self.psf, rot_mat.T)[0])
        #psf_rotated_fft = rotation(fft.fftn(self.psf), rot_mat.T)[0]
        #psf_rotated_fft = fft.fftn(rotated_PSF)
        if known_trans:
            view_rotated_fft = fft.fftn(rotation(view, rot_mat.T, trans_vec=-rot_mat.T@np.array(trans_vec), order=interp_order)[0])
            #view_rotated_fft = rotation(fft.fftn(view), rot_mat.T, trans_vec=-rot_mat.T@trans_vec)[0]
        else:
            view_rotated_fft = fft.fftn(rotation(view, rot_mat.T, order=interp_order)[0])
            #view_rotated_fft = rotation(fft.fftn(view), rot_mat.T, order=interp_order)[0]
            shift, _, _ = phase_cross_correlation(psf_rotated_fft*self.volume_fourier, view_rotated_fft, space='fourier', upsample_factor=10, normalization=None)
            #psf_rotated_fft = fourier_shift(psf_rotated_fft, rot_mat@shift)
            view_rotated_fft = fourier_shift(view_rotated_fft, shift)
            if save_shift:
                recorded_shifts[view_idx].append(-rot_mat@shift)
        energy = np.linalg.norm(psf_rotated_fft*self.volume_fourier - view_rotated_fft)**2/(self.size**self.nb_dim)
        variables_used_to_compute_gradient = [psf_rotated_fft, view_rotated_fft]
        return energy, variables_used_to_compute_gradient
    
    def get_energy_gpu(self, rot_mat, trans_vec, view, recorded_shifts, view_idx, save_shift=False, known_trans=True, interp_order=3):
        psf_rotated_fft = cp.fft.fftn(rotation_gpu(self.psf_gpu, rot_mat.T)[0])
        if known_trans:

            view_rotated_fft = cp.fft.fftn(rotation_gpu(view, rot_mat.T, trans_vec=-rot_mat.T@trans_vec, order=interp_order)[0])

        else:
            rot_im = rotation_gpu(view, rot_mat.T, order=interp_order)[0]
            view_rotated_fft = cp.fft.fftn(rot_im)
            # print('psf shape', psf_rotated_fft.shape)
            shift, _, _ = phase_cross_correlation_gpu(psf_rotated_fft*self.volume_fourier_gpu, view_rotated_fft, space='fourier', upsample_factor=10, normalization=None)

            view_rotated_fft = fourier_shift_gpu(view_rotated_fft, shift)
            if save_shift:
                recorded_shifts[view_idx].append(-rot_mat@shift)
        energy = cp.linalg.norm(psf_rotated_fft*self.volume_fourier_gpu - view_rotated_fft)**2/(self.size**self.nb_dim)
        variables_used_to_compute_gradient = [psf_rotated_fft, view_rotated_fft]
        return energy, variables_used_to_compute_gradient

    def get_energy_2(self, rot_mat, trans_vec, view_fft, known_trans=True):
        rotated_volume = rotation(self.volume_fourier, rot_mat)[0]
        convolved_rotated_volume = self.psf_fft * rotated_volume
        if not known_trans:
            shift, _, _ = phase_cross_correlation(convolved_rotated_volume, view_fft)
        else:
            shift = trans_vec
        translated_convolved_rotated_volume = fourier_shift(convolved_rotated_volume, shift)
        energy = np.linalg.norm(translated_convolved_rotated_volume-view_fft)**2/(self.size*self.nb_dim)
        return energy

    def one_gd_step(self, rot_mat, trans_vec, views, lr, known_trans, view_idx, recorded_shifts, reg_coeff=0, ground_truth=None, interp_order=3):
        view = views[view_idx]
        try:
            tr = trans_vec.get()
        except:
            tr = trans_vec
        energy, variables_used_to_compute_gradient = \
            self.get_energy(rot_mat, tr, view, recorded_shifts, view_idx, save_shift=True,
                            known_trans=known_trans, interp_order=interp_order)
        psf_rotated_fft, view_rotated_fft = variables_used_to_compute_gradient
        grad = psf_rotated_fft * (psf_rotated_fft*self.volume_fourier-view_rotated_fft)
        self.volume_fourier -= lr*grad
        #gradient_l2_reg, l2_reg  = self.l2_regularization()
        #self.volume_fourier -= lr*reg_coeff*gradient_l2_reg
        self.volume_fourier_gpu = cp.array(self.volume_fourier)
        if not known_trans and ground_truth is not None:
            # _, self.volume_fourier = shift_registration_exhaustive_search(np.fft.fftn(ground_truth), self.volume_fourier, fourier_space=True)
            #self.volume_fourier = center_barycenter(self.volume_fourier)
            pass
        return energy

    def squared_variations_regularization(self):
        v1 = self.volume_fourier[1:, :, :]
        v2 = self.volume_fourier[:, 1:, :]
        v3 = self.volume_fourier[:, :, 1:]
        v1 = np.pad(v1, ((0,1), (0,0), (0,0)))
        v2 = np.pad(v2, ((0,0), (0,1), (0,0)))
        v3 = np.pad(v3, ((0,0), (0,0), (0,1)))
        TV = np.mean((self.volume_fourier-v1)**2) + np.mean((self.volume_fourier-v2)**2) + np.mean((self.volume_fourier-v3)**2)
        gradient_TV = 2*self.volume_fourier*(3*self.volume_fourier - v1 - v2 - v3)/self.size**3
        return gradient_TV,np.real(TV)

    def l2_regularization(self):
        gradient_l2_reg = 2*self.volume_fourier
        l2_reg = np.mean(np.abs(self.volume_fourier)**2)
        return gradient_l2_reg, l2_reg

    def get_image_from_fourier_representation(self):
        ifft = fft.ifftn(self.volume_fourier)
        image = np.abs(fft.fftshift(ifft))
        return image

    def save(self, output_dir, output_name):
        path = f'{output_dir}/{output_name}.tif'
        im = self.get_image_from_fourier_representation()
        save(path, im)
        return im

    def register_and_save(self, output_dir, output_name, ground_truth_path=None, translate=False):
        path = f'{output_dir}/{output_name}.tif'
        im = self.get_image_from_fourier_representation()
        if translate:
            im = translate_to_have_one_connected_component(im)
        if ground_truth_path is not None and translate:
            _, im = shift_registration_exhaustive_search(read_image(ground_truth_path), im)
        save(path, im)
        if ground_truth_path is not None:
            gt = read_image(ground_truth_path)
            _, registered_array = registration_exhaustive_search(gt, read_image(path), output_dir, f'{output_name}_registered', self.nb_dim)
            _, im_registered = registration_exhaustive_search(gt, registered_array, output_dir, f'recons_registered_gradient', 3,
                                           sample_per_axis=40, gradient_descent=True)
            return im_registered



if __name__ == '__main__':

    from scipy import ndimage, misc
    import matplotlib.pyplot as plt
    import numpy.fft

    from skimage.registration import phase_cross_correlation





    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()  # show the filtered result in grayscale
    ascent = misc.ascent()
    input_ = numpy.fft.fft2(ascent)
    result = ndimage.fourier_shift(input_, shift=200)
    result = numpy.fft.ifft2(result)
    result = np.abs(fft.fftshift(result))

    ax1.imshow(ascent)
    ax2.imshow(result)  # the imaginary part is an artifact
    plt.show()
