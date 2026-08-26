"""

from utils import resize

from utils import rotation
from GMM.GMM import *
from pyfigtree import figtree
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from time import time
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import *
from common_image_processing_methods.rotation_translation import *
from volume_representation.pixel_representation import *
from manage_matplotlib.graph_setup import *
from classes_with_parameters import ParametersDataGeneration
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import nd_gaussian_with_points
from manage_matplotlib.graph_setup import set_up_graph
from scipy.ndimage import fourier_shift


sz = 27
def find_complexity(values_to_test, var_to_test_name, nb_tests, f, title, xlabel, extra_label='', show=True, **f_args):
    set_up_graph(MEDIUM_SIZE=sz, SMALLER_SIZE=sz)
    times = np.zeros(len(values_to_test))
    for i, size in enumerate(values_to_test):
        print('size', size)
        f_args[var_to_test_name] = values_to_test[i]
        avg_time = 0
        for t in range(nb_tests):
            ex_time = f(**f_args)
            avg_time += ex_time
        avg_time /= nb_tests
        times[i] = avg_time
    log_values = np.log(values_to_test)
    log_times_mean = np.log(times)
    plt.scatter(log_values, log_times_mean, marker='X', color='red', label="points experimentaux")
    reg = LinearRegression().fit(np.expand_dims(log_values, 1), log_times_mean)

    score = reg.score(np.expand_dims(log_values, 1), log_times_mean)
    plt.plot(log_values, reg.predict(np.expand_dims(log_values, 1)),
             label=f"{extra_label} modèle linéaire : coefficient directeur {round(reg.coef_[0], 2)}, \n "
                   f"ordonnée à l'origine : {round(reg.intercept_, 2)}, "
                   f"score : {round(score, 2)}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("logarithme du temps d'évaluation (s)")
    plt.grid()
    if show:
        plt.legend()
        plt.show()

"""
def computation_time_two_by_two_distances(nb_points):
    points = np.random.randint(0, 10, (nb_points, 3))
    t = time()
    dist = compute_two_by_two_distances(points)
    computation_time = time() - t
    return computation_time


list_nb_points = [300, 400, 500, 600,700,800,900,1000, 1100, 1200, 1300]
find_complexity(list_nb_points, 'nb_points',1, computation_time_two_by_two_distances, 'computation time two by two distances between points', 'number of points')




def time_gmm_figtree(nb_gaussians, size, sigma):
    grid = make_grid(size, 3)
    grid_step = 2 / (size - 1)
    grid_flatten = grid.reshape((size ** 3, 3))
    points = 2 * np.random.random(size=(nb_gaussians, 3)) - 1
    coeffs = np.ones(nb_gaussians)
    t = time()
    figtree(points, grid_flatten, coeffs, bandwidth=1.5 * sigma * grid_step)
    return time() - t
"""

# image_size()
from scipy.ndimage import fourier_shift


def time_one_fourier_shift(size):
    im = np.random.random((size, size, size))
    shift = 10*np.random.random(3)
    t = time()
    fourier_shift(im, shift)
    return time() - t

def time_one_rotation(size):
    im_to_rotate = np.random.random((size, size, size))
    rot_mat = get_3d_rotation_matrix([45, 20, 30])
    t = time()
    rotated, _ = rotation(im_to_rotate, rot_mat)
    return time() - t


def time_one_fft(size):
    im = np.random.random((size, size, size))
    t = time()
    np.fft.fftn(im)
    return time() - t


def time_one_phase_corr(size, space='fourier'):
    im1 = np.random.random((size, size, size))
    im2 = np.random.random((size, size, size))
    t0 = time()
    phase_cross_correlation(im1, im2, space=space)
    t1 = time() - t0
    # print('time whole', t1)
    return t1


def rotation_comlexity(sizes_to_test, nb_tests):
    times = np.zeros(len(sizes_to_test))
    for i, size in enumerate(sizes_to_test):
        avg_time = 0
        for t in range(nb_tests):
            im_to_rotate = np.random.random((size, size, size))
            t = time()
            rotated, _ = rotation(im_to_rotate, [45, 20, 30])
            avg_time += (time() - t)
        avg_time /= nb_tests
        times[i] = avg_time
    find_complexity(sizes_to_test, times, 'rotation 3D : analyse de la complexité',
                    "logarithme de la taille d'un côté du volume (pixels)")
    # print(temps_rotation/temps_mixture)
    # print('temps rot', temps_rotation)
    # print('temps gaussian', temps_mixture)


def time_one_gaussian(sigma, size, nb_gaussians, nb_dim, cov_psf):
    grid = make_grid(size, 3)
    grid_step = 2 / (size - 1)
    coeffs = np.array([1] * nb_gaussians)
    # centers = 2*(np.random.random((nb_gaussians, nb_dim))-0.5)
    centers = np.zeros((nb_gaussians, nb_dim))
    cov = sigma ** 2 * grid_step ** 2 * np.eye(nb_dim) + cov_psf
    covs = np.array([cov] * nb_gaussians)
    t = time()
    gm, _ = gaussian_mixture(grid, coeffs, centers, covs, nb_dim, 3)
    temps_one_g = (time() - t) / nb_gaussians
    return temps_one_g


def time_N_gaussians(nb_gaussians, sigma, size, nb_dim):
    grid = make_grid(size, 3)
    grid_step = 2 / (size - 1)
    coeffs = np.array([1] * nb_gaussians)
    # centers = 2*(np.random.random((nb_gaussians, nb_dim))-0.5)
    centers = np.zeros((nb_gaussians, nb_dim))
    cov = sigma ** 2 * grid_step ** 2 * np.eye(nb_dim)
    covs = np.array([cov] * nb_gaussians)
    t = time()
    gm, _ = gaussian_mixture(grid, coeffs, centers, covs, nb_dim, 3)
    return time() - t


def time_gaussians_evaluation(nb_points):
    nb_dim = 3
    points = 2*np.random.random((nb_points, nb_dim)) - 1
    t = time()
    gauss = nd_gaussian_with_points(points, 2*np.random.random(nb_dim)-1, 10**-2 * np.eye(nb_dim))
    return time() - t


def time_one_multiplication(size):
    arr = np.random.random((size,3))
    rot_mat = get_3d_rotation_matrix([21,14,24], "zxz")
    t = time()
    print('shape', arr.shape)
    print('sg', rot_mat.shape)
    res = arr @ rot_mat
    t2 = time() -t
    return t2


sigma_def = 2
size_def = 50
sizes = np.arange(50,150,5)


find_complexity(sizes, 'size', 50, time_one_fft,'',
                "logarithme de la taille d'un côté du volume (pixels)")

1/0

find_complexity(sizes, 'size', 50, time_one_fourier_shift, "",
                "logarithme de de la taille d'un coté du volume (pixels)")
1/0



1/0



find_complexity(np.arange(2000, 5000,100), 'nb_points', 100,
                time_gaussians_evaluation, "","logarithme du nombre de points de la grille")
1/0

"""
sigma_def = 2
size_def = 50
nb_gaussians_def = 50
sigmas_pixel = np.arange(1.5, 7, 0.5)

find_complexity([20,25,30,35,40,45,50,55,60,65,70], 'size', 200, time_one_phase_corr, "Temps d'une correlation de phase en fonction de la taille de l'image",
                "logarithme de de la taille d'un coté de l'image")
1/0

find_complexity([20,25,30,35,40,45,50,55,60,65,70], 'size', 100, time_one_rotation, "Temps d'une rotation en fonction de la taille de l'image",
                "logarithme de de la taille d'un coté de l'image")

find_complexity([20,25,30,35,40,45,50,55,60,65,70], 'size', 100, time_one_rotation, "Temps d'une rotation en fonction de la taille de l'image",
                "logarithme de de la taille d'un coté de l'image")


find_complexity(sigmas_pixel, 'sigma', 50, time_one_gaussian,
                "écart type des gaussiennes : analyse de la complexité", "logarithme de l'écart type des gaussiennes",
                size=size_def, nb_gaussians=nb_gaussians_def, nb_dim=3)

1/0
"""
sizes = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
nb_gaussianss = [10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100, 150, 200]

find_complexity(nb_gaussianss, 'nb_gaussians', 20, time_N_gaussians,
                "temps par gaussiennes : analyse de la complexité (écart type de 2 pixels)",
                "logarithme du nombre de gaussiennes",
                sigma=sigma_def, size=size_def, nb_dim=3)

1/0

for size in [10, 20, 30, 40, 50, 60]:
    find_complexity(nb_gaussianss, 'nb_gaussians', 10, time_gmm_figtree,
                    f"nombre de gaussiennes : analyse de la complexité de figtree, sigma : {sigma_def}",
                    "logarithme du nombre de gaussiennes",
                    f'size : {size}, ', show=False,
                    sigma=sigma_def, size=size)
plt.legend()
plt.grid()
plt.show()

for sigma in [2, 3, 4, 5, 6]:
    find_complexity(nb_gaussianss, 'nb_gaussians', 10, time_gmm_figtree,
                    f"nombre de gaussiennes : analyse de la complexité de figtree, taille d'un côté : {size_def}",
                    "logarithme du nombre de gaussiennes",
                    f'sigma : {sigma}, ', show=False,
                    sigma=sigma, size=size_def)
plt.legend()
plt.show()

find_complexity(sizes, 'size', 50, time_one_gaussian,
                "taille du volume, analyse de la complexité d'évaluation d'une GMM",
                "logarithme de la taille d'un côté du volume (pixels)",
                sigma=sigma_def, nb_gaussians=nb_gaussians_def, nb_dim=3)

find_complexity(nb_gaussianss, 'nb_gaussians', 20, time_N_gaussians,
                "temps par gaussiennes : analyse de la complexité (écart type de 2 pixels)",
                "logarithme du nombre de gaussiennes",
                sigma=sigma_def, size=size_def, nb_dim=3)



# rotation_comlexity(sizes, 5)
# fourrier_transform_complexity(sizes, 100)
# complexity_phase_correlation([20,25,30,35,40,45,50,60,70,80,90], 100)

find_complexity(sizes, 'size', 20, time_one_rotation, 'rotation 3D : analyse de la complexité',
                "logarithme de la taille d'un côté du volume (pixels)")
find_complexity(sizes, 'size', 20, time_one_fft, 'transformée de Fourier 3D : analyse de la complexité',
                "logarithme de la taille d'un côté du volume (pixels)")
