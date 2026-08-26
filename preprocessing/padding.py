from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import make_grid
import numpy as np
from common_image_processing_methods.otsu_thresholding import otsu_thresholding
from common_image_processing_methods.others import resize


def pad_im_to_include_sphere_containing_object(im, h=0):
    center_sphere, diameter_sphere = find_sphere_containing_object(im)
    print('sphere center', center_sphere)
    print('diameter sphere', diameter_sphere)
    pad_width = np.max([diameter_sphere/2 - center_sphere[d] for d in range(len(center_sphere))] +
                       [center_sphere[d]+diameter_sphere/2 - im.shape[0] for d in range(len(im.shape))])
    pad_width += h
    pad_width = int(pad_width)+1
    print('pad width', pad_width)
    if pad_width >= 0:
        im_padded = np.pad(im, pad_width)
    else:
        im_size = im.shape[0]
        im_padded = im[-pad_width:im_size+pad_width, -pad_width:im_size+pad_width,-pad_width:im_size+pad_width]
    return im_padded


def compute_two_by_two_distances(points):
    """
    points : array containing a set of point, shape (Nb_point, nb_dim)
    compute matricially 2 by 2 distances between points"""
    points_repeated = np.repeat(np.expand_dims(points,axis=0), len(points), axis=0)
    points_repeated_transpose = np.transpose(points_repeated, (1,0,2))
    diff = points_repeated - points_repeated_transpose
    squared_2_by_2_distances = np.sum(diff**2, axis=2)
    return squared_2_by_2_distances


def find_support(im):
    grid = make_grid(im.shape[0], len(im.shape), mi=None, ma=None)
    support = grid[im>0]
    return support


def find_sphere_containing_object(im, size=30):
    size_before = im.shape[0]
    im_resized = resize(im, [size for _ in range(len(im.shape))])
    im_resized = otsu_thresholding(im_resized, 5)
    support = find_support(im_resized)
    squared_two_by_two_distances = compute_two_by_two_distances(support)
    argma_indices = np.unravel_index(squared_two_by_two_distances.argmax(), squared_two_by_two_distances.shape)
    center_sphere = (support[argma_indices[0]] + support[argma_indices[1]])/2
    diameter_sphere = np.sqrt(squared_two_by_two_distances[argma_indices])
    diameter_sphere *= (size_before/size)
    center_sphere *= (size_before/size)
    return center_sphere, diameter_sphere