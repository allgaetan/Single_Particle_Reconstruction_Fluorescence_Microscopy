from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import gaussian_mixture, make_grid
from common_image_processing_methods.otsu_thresholding import otsu_thresholding
import numpy as np


def simulate_partial_labelling(image, nb_whole=120, std_min=0.02, std_max=0.05, coeff_min=0, coeff_max=1):
    """simulates partial labelling by substracting gaussians to the image at random positions
    nb_whole : number of gaussians substracted
    standard deviations of gaussians are drawn unioformly between std_min and std_max. (standard deviation are expressed in proportion of
    image size)
    coefficient of gaussians are drawn uniformly between coeff_min and coeff_max
    """
    nb_dim = len(image.shape)
    grid = make_grid(image.shape[0], nb_dim)
    image_thresh = otsu_thresholding(image, sig_blur=5)
    non_zeros_pixel_cooridnates = grid[image_thresh>0]
    non_zeros_pixels_vals = image_thresh[image_thresh>0]
    centers_whole_indices = np.random.randint(0,len(non_zeros_pixel_cooridnates), nb_whole)
    centers_whole = non_zeros_pixel_cooridnates[centers_whole_indices]
    sigmas_whole = (std_max-std_min) * np.random.random(nb_whole) + std_min
    coeffs_whole = (coeff_max - coeff_min) * np.random.random(nb_whole) + coeff_min
    coeffs_whole = np.array([min(coeffs_whole[i], non_zeros_pixels_vals[centers_whole_indices[i]]) for i in range(len(coeffs_whole))])
    covs = [(2*sig)**2*np.eye(nb_dim) for sig in sigmas_whole]
    wholes, _ = gaussian_mixture(grid, coeffs_whole, centers_whole, covs, nb_dim, 3)
    image_with_whole = image - wholes
    image_with_whole[image_with_whole<0] = 0
    return image_with_whole


if __name__ == '__main__':
    from skimage import io
    im_path = "../../results/synthetic_data/recepteurs_AMPA/2D/ground_truth.tif"
    im = io.imread(im_path)
    image_thresh = otsu_thresholding(im, sig_blur=5)
    grid = make_grid(im.shape[0], 2, mi=None, ma=None)
    non_zeros_pixel_cooridnates = grid[image_thresh > 0]
    a = 1


    # image_with_whole = simulate_partial_labelling(im, 50,0.01,0.03,0,1)
    # save("im_whole.tif", image_with_whole)






