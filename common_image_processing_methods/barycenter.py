import numpy as np
from common_image_processing_methods.rotation_translation import translation
from scipy.ndimage import gaussian_filter
import copy as cp

def make_grid_2(shape, nb_dim):
    slices = []
    for d in range(nb_dim):
        slices.append(slice(0,shape[d], 1))
    slices = tuple(slices)
    transpose_idx = list(range(1,nb_dim+1))
    transpose_idx.append(0)
    grid = np.mgrid[slices].transpose(*transpose_idx)
    return grid


def compute_barycenter(image):
    """compute barycenter coordinates of object"""
    grid = make_grid_2(image.shape, len(image.shape))
    axs = list(range(len(image.shape)))
    weighet_sum = np.tensordot(grid, image, (axs, axs))
    return weighet_sum/np.sum(image)


def center_barycenter(image, sigma_filtered=None, thersh=0.5):
    """center barucenter of object at image center"""
    if sigma_filtered is not None:
        image_filtered = gaussian_filter(image,sigma_filtered)
    else:
        image_filtered = cp.deepcopy(image)
    image_filtered[image_filtered<=thersh * np.max(image_filtered)] = 0
    barycenter = compute_barycenter(image_filtered)
    image_center = np.array([image.shape[d]//2 for d in range(len(image.shape))])
    trans_vec = image_center - barycenter
    translated_image = translation(image, trans_vec)
    return translated_image, trans_vec


def center_barycenter_4d(image_4d, sigma_filtered=None, thersh=0.5):
    id = 0
    _, trans_vec = center_barycenter(image_4d[id], sigma_filtered, thersh)
    im_out = []
    for i in range(image_4d.shape[0]):
        im = image_4d[i]
        translated_im = translation(im, trans_vec)
        im_out.append(translated_im)
    return np.array(im_out)
