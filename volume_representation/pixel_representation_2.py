import numpy as np
from numpy import fft
from common_image_processing_methods.rotation_translation import rotation, translation
from manage_files.read_save_files import save, read_image
from common_image_processing_methods.registration import registration_exhaustive_search
from skimage.registration import phase_cross_correlation
from scipy.ndimage.fourier import fourier_shift
from common_image_processing_methods.others import normalize, crop_center
from common_image_processing_methods.registration import translate_to_have_one_connected_component
from data_generation.generate_data import get_PSF_with_stds


def phase_cross_corralation_several_channels(reference_images, moving_images, space='fourier'):
    # assume complex data is already in Fourier space
    if space == 'fourier':
        srcs_freq = reference_images
        targets_freq = moving_images
    # real data needs to be fft'd.
    elif space == 'real':
        srcs_freq = [fft.fftn(reference_image) for reference_image in reference_images]
        targets_freq = [fft.fftn(moving_image) for moving_image in moving_images]
    else:
        raise ValueError('space argument must be "real" of "fourier"')
    shape = srcs_freq[0].shape
    nb_channels = len(reference_images)
    cross_corelation_sum = np.zeros(shape).astype(dtype=np.complex_)
    for c in range(nb_channels):
        image_product = srcs_freq[c] * targets_freq[c].conj()
        cross_correlation = fft.ifftn(image_product)
        cross_corelation_sum += cross_correlation

    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_corelation_sum)),
                              cross_corelation_sum.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.stack(maxima).astype(np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    return shifts


class Fourier_pixel_representation:
    def __init__(self, nb_dim, size, psfs, sigma_z_psf=None, init_vol_path=None):
        nb_channels = len(psfs)
        self.volume_fourier = [np.random.random(tuple([size for _ in range(nb_dim)])).astype(dtype=np.complex_) for c in range(nb_channels)]
        self.nb_dim = nb_dim
        self.size = size
        self.psfs = psfs
        self.nb_channels = nb_channels
        if sigma_z_psf is not None:
            self.crop_size_psf = min(int(6*sigma_z_psf), size)
        else:
            self.crop_size_psf = size

    def get_energy(self, rot_mat, trans_vec, view, recorded_shifts, view_idx, save_shift=False, known_trans=True):
        nb_channels = view.shape[0]
        rot_mat = rot_mat.T
        #size_crop = 10
        #rotated_PSF = crop_center(rotation(crop_center(self.p00sf, (self.crop_size_psf, self.crop_size_psf, self.crop_size_psf)), rot_mat)[0], self.psf.shape)
        psf_rotated_fft = [fft.fftn(rotation(self.psfs[c], rot_mat)[0]) for  c in range(nb_channels)]
        #psf_rotated_fft = fft.fftn(rotated_PSF)
        if known_trans:
            view_rotated_fft = [fft.fftn(rotation(view[c], rot_mat, trans_vec=-rot_mat@trans_vec)[0]) for c in range(nb_channels)]
        else:
            view_rotated_fft = [fft.fftn(rotation(view[c], rot_mat)[0]) for c in range(nb_channels)]
            shift, _, _ = phase_cross_corralation_several_channels([psf_rotated_fft[c]*self.volume_fourier[c] for c in range(nb_channels)],
                                                                   view_rotated_fft, space='fourier')
            view_rotated_fft = [fourier_shift(view_rotated_fft[c], shift) for c in range(nb_channels)]
        energy = np.sum([np.linalg.norm(psf_rotated_fft[c]*self.volume_fourier[c] - view_rotated_fft[c])**2/(self.size**self.nb_dim) for c in range(nb_channels)])
        variables_used_to_compute_gradient = [psf_rotated_fft, view_rotated_fft]
        return energy, variables_used_to_compute_gradient

    def one_gd_step(self, rot_mat, trans_vec, views, lr, known_trans, view_idx, recorded_shifts, reg_coeff=0):
        view = views[view_idx]

        nb_channels = view.shape[0]
        energy, variables_used_to_compute_gradient = \
            self.get_energy(rot_mat, trans_vec, view, recorded_shifts, view_idx, save_shift=True,
                            known_trans=known_trans)
        psf_rotated_fft, view_rotated_fft = variables_used_to_compute_gradient
        for c in range(nb_channels):
            grad = psf_rotated_fft[c] * (psf_rotated_fft[c]*self.volume_fourier[c]-view_rotated_fft[c])
            self.volume_fourier[c] -= lr*grad
        # TV, gradient_TV = self.squared_variations_regularization()
        # self.volume_fourier -= lr*gradient_TV
        return energy

    def squared_variations_regularization(self):
        v1 = self.volume_fourier[1:, :, :]
        v2 = self.volume_fourier[:, 1:, :]
        v3 = self.volume_fourier[:, :, 1:]
        v1 = np.pad(v1, ((0,1), (0,0), (0,0)))
        v2 = np.pad(v2, ((0,0), (0,1), (0,0)))
        v3 = np.pad(v3, ((0, 0), (0, 0), (0,1)))
        TV = np.mean((self.volume_fourier-v1)**2) + np.mean((self.volume_fourier-v2)**2) + np.mean((self.volume_fourier-v3)**2)
        gradient_TV = 2*self.volume_fourier*(3*self.volume_fourier - v1 - v2 - v3)/self.size**3
        return TV, gradient_TV

    def get_image_from_fourier_representation(self):
        images = []
        for c in range(self.nb_channels):
            ifft = fft.ifftn(self.volume_fourier)
            image = np.abs(fft.fftshift(ifft))
            images.append(image)
        return images

    def register_and_save(self, output_dir, output_name, ground_truth_path=None, translate=False):

        ims = self.get_image_from_fourier_representation()
        if translate:
            for c in range(self.nb_channels):
                path = f'{output_dir}/{output_name}_channel_{c}.tif'
                im_translated = translate_to_have_one_connected_component(ims[c])
                save(path, im_translated)
        #if ground_truth_path is not None:
            #registration_exhaustive_search(ground_truth_path, path, output_dir, f'{output_name}_registered', self.nb_dim)




if __name__ == '__main__':

    from scipy import ndimage, misc
    import matplotlib.pyplot as plt
    import numpy.fft
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