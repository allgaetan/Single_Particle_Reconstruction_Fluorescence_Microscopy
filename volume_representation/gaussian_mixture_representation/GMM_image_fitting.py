import matplotlib.pyplot as plt
from utils import *
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import *

import numpy as np

from skimage import io
from skimage.metrics import structural_similarity as ssim
from volume_representation.gaussian_mixture_representation.GMM_representation import GMM_representation
from learning_algorithms.gradient_descent_known_angles import gradient_descent_known_rot
from classes_with_parameters import ParametersMainAlg


def read_views(fold_name, nb_dim, desired_size):
    views = []
    rot_vecs = []
    trans_vecs = []
    for fn in os.listdir(fold_name):
        view = io.imread(f'{fold_name}/{fn}')
        view = resize(view, tuple(desired_size for _ in range(nb_dim)))
        # view = normalize(view)
        splitted_name = fn.split('_')
        if nb_dim == 3:
            t0,t1,t2 = float(splitted_name[-4]), float(splitted_name[-3]), float(splitted_name[-2])
            r0,r1,r2 = float(splitted_name[-8]), float(splitted_name[-7]), float(splitted_name[-6])
            rot_vecs.append([r0, r1, r2])
            trans_vecs.append([t0,t1,t2])
        else:
            angle = splitted_name[-5]
            angle = float(angle)
            rot_vecs.append(angle)
            t1 = splitted_name[-3]
            t2 = splitted_name[-2]
            trans_vecs.append([t1,t2])
        views.append(view)

    views = np.array(views)
    rot_vecs = np.array(rot_vecs)
    trans_vecs = np.array(trans_vecs)
    if len(views) == 0:
        print(f'WARNING : no view was read in {fold_name}, check the name of the folder')
    return views, rot_vecs, trans_vecs


def find_nb_gaussians_given_sigma(image, cov, size):
    # image = normalize(resize(image, size))
    grid_step = 2/(size-1)
    nb_dim = len(image.shape)
    """
    print('cov', cov)
    grid = make_grid(size, nb_dim)
    GM, _ = gaussian_mixture(grid, np.array([1]), np.array([[0,0,0]]), [cov], nb_dim, 3)
    print('sum GM', np.sum(GM))
    """
    volume_one_gaussian = (np.sqrt(2*np.pi))**nb_dim * np.sqrt(cov).diagonal().prod()/grid_step**nb_dim
    volume_image = np.sum(image)
    #print('sum im', volume_image)
    #print('volume one gauss', volume_one_gaussian)
    nb_gaussians = 2*volume_image/volume_one_gaussian
    return int(nb_gaussians+1)




def arg_max_array(arr):
    arg_ma = np.unravel_index(arr.argmax(), arr.shape)
    ma = arr[arg_ma]
    return np.array(arg_ma), ma


def find_nb_gaussians(views, size, cov_PSF, sigma, lr, eps_gd, nb_gaussians_range, nb_tests=5):
    nb_dim = len(views[0].shape)

    dists_to_gt = []
    grid = make_grid(size, nb_dim)
    grid_step = 2/(len(grid)-1)
    cov = sigma ** 2 * np.eye(nb_dim) + grid_step**2*cov_PSF
    for nb_gaussians in nb_gaussians_range:
        avg_dist_to_gt = 0
        for v in range(len(views)):
            for _ in range(nb_tests):
                gd_obj = GMM_gradient_descent_grid(np.array([views[v]]), np.array([np.eye(nb_dim)]), np.zeros((1, nb_dim)), '',
                                                   nb_gaussians, nb_dim, size,
                                                   lr, cov_PSF, 3, 0, None, False, True, False,
                                                   True, True, False)
                centers, _, coeffs = gd_obj.random_init(sigma)
                centers, _, coeffs, _ = gd_obj.gradient_descent(np.array(centers), np.array([sigma] * len(centers)),
                                                                np.array(coeffs), 100, eps_gd, views[v])
                covs = [cov for _ in range(len(coeffs))]
                GM, _ = gaussian_mixture(grid, np.array(coeffs), np.array(centers), covs, nb_dim, 3)
                dist_to_gt = np.sum((GM - views[v]) ** 2)
                print('dist to gt',dist_to_gt)
                avg_dist_to_gt += dist_to_gt
        avg_dist_to_gt/=(len(views)*nb_tests)
        dists_to_gt.append(avg_dist_to_gt)
    plt.plot(nb_gaussians_range, dists_to_gt)
    plt.show()


class Gmm_rep_of_view:
    def __init__(self, view_voxel, sigma=0.03, cov_PSF=np.array([[25 , 0, 0],
                                                                    [0, 1, 0],
                                                                    [0, 0, 1]]), nb_gaussians=None):
        self.view_voxel = view_voxel
        size = view_voxel.shape[0]
        nb_dim = 3
        grid_step = 2 / (size - 1)
        cov_PSF = grid_step ** 2 * cov_PSF
        cov = sigma ** 2 * np.eye(nb_dim) + cov_PSF
        GM, coeffs, centers, nb_gaussians = heuristic_gmm_fitting(view_voxel, sigma, cov_PSF, nb_gaussians)
        self.GM = GM
        self.coeffs = coeffs
        self.centers = centers
        self.nb_gauss = nb_gaussians

    def save(self, path):
        save(path, self.GM)


def heuristic_gmm_fitting(image, sigma, cov_PSF,nb_gaussians=None):
    cov = sigma ** 2 * np.eye(3) + cov_PSF
    size = image.shape[0]
    nb_dim = len(image.shape)
    grid = make_grid(size, nb_dim)
    grid_step = 2 / (size - 1)
    coeffs = []
    centers = []
    GM = np.zeros(image.shape)
    if nb_gaussians is None:
        opt_nb_gaussian = find_nb_gaussians_given_sigma(image, cov, size)
        opt_nb_gaussian += 10
        nb_gaussians = opt_nb_gaussian
    print('nb gaussians', nb_gaussians)
    # nb_gaussians = 72
    # nb_non_zeros_pixels = len(image[image > 10 ** -2])
    for _ in range(nb_gaussians):
        new_center, coeff = arg_max_array(image - GM)
        new_center = grid_step * (new_center - size + 1) + 1
        centers.append(new_center)
        coeffs.append(coeff)
        covs = [cov for _ in range(len(coeffs))]
        GM, _ = gaussian_mixture(grid, coeffs, centers, covs, nb_dim, 3)


    print('centres', len(centers))
    gm_rep = GMM_representation(nb_gaussians, sigma, 3, size, cov_PSF, 0)
    gm_rep.centers = np.array(centers)
    gm_rep.coeffs = np.array(coeffs)
    gradient_descent_known_rot(gm_rep, np.zeros((1,3)), np.zeros((1,3)), [image], ParametersMainAlg(lr=10**-4, N_iter_max=30), True)
    covs = [cov for _ in range(len(coeffs))]
    GM, _ = gaussian_mixture(grid, gm_rep.coeffs, gm_rep.centers, covs, nb_dim, 3)
    """
    gd_obj = GMM_gradient_descent_grid(np.array([image]), np.array([np.eye(nb_dim)]), np.zeros((1, nb_dim)), np.zeros((1, nb_dim)), '', [sigma]*len(centers), nb_dim, size,
                                       lr, cov_PSF/grid_step**2, 3, 0, False, False, False, None, 0, 0)
    # centers, _, coeffs = gd_obj.random_init(sigma)
    centers, coeffs, _, _ = gd_obj.gradient_descent(np.array(centers),np.array(coeffs), 100, eps)
    covs = [cov for _ in range(len(coeffs))]
    GM, _ = gaussian_mixture(grid, np.array(coeffs), np.array(centers), covs, nb_dim, 3)
    ssim_gt = ssim(GM, image)
    """

    return GM, coeffs, centers, nb_gaussians


if __name__ == '__main__':
    from manage_files.paths import PTH_GT, PATH_PROJECT_FOLDER
    from classes_with_parameters import ParametersDataGeneration
    #pth_im = f'{PATH_PROJECT_FOLDER}/recepteurs_AMPA.tif'
    pth_im = f'{PTH_GT}/recepteurs_AMPA.tif'
    pth_im = f'{PATH_PROJECT_FOLDER}/views/recepteurs_AMPA/single_view/sig_z_5/view_12.tif'
    params_data_gen = ParametersDataGeneration(nb_views=10)
    cov_PSF = params_data_gen.get_cov_psf()
    print('cov psf', cov_PSF)
    im = read_image(pth_im)
    a = find_nb_gaussians_given_sigma(im, 0.03 ** 2 * np.eye(3)  , 50)
    print('a', a)
    1/0


    im = read_image(pth_im)
    """
    sigma = 0.0728
    cov = sigma ** 2 * np.eye(3)
    
    GM, _, _, _ = heuristic_gmm_fitting(im, cov, 10**-4, 0.1)
    """
    view = Gmm_rep_of_view(im)
    view.save(f'{PATH_PROJECT_FOLDER}/view_gmm.tif')
    #print('GM', np.max(GM))
    #save(f'{PATH_PROJECT_FOLDER}/ampa_fitted.tif', GM)
    1/0

    #pth = 'tests_on_particules/emd_0427/2D/4_views_10/views/view_144_.tif'
    #pth = 'tests_on_particules/synthetic_centriole/2D/top_view/9_views_10/views/view_17_.tif'
    #pth = 'tests_on_particules/emd_0427/2D/ground_truth_0_.tif'
    #pth = '../tests_on_particules/emd_0427/3D/ground_truth.tif'
    pth = '../tests_on_particules/synthetic_centriole/3D/ground_truth.tif'
    pth_views = '../tests_on_particules/emd_0427/3D/12_views_5/views'
    image = io.imread(pth)
    cov_PSF = np.array([[25 , 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

    sigma = 0.0728
    size = 50
    views, _, _ = read_views(pth_views, 3, size)
    print('sum views', [np.sum(views[v]) for v in range(len(views))])
    print('sum gt', np.sum(image))

    list_nb_gaussians = [5,10,20,30,50,60,70, 80,85,87,90,92,95,97,100,110,120,130,140,150]
    # list_nb_gaussians = [20, 60]
    ssims_gt = np.zeros(len(list_nb_gaussians))
    ssims_views = np.zeros((len(list_nb_gaussians), len(views)))
    list_nb_zeros_coeffs = []
    for i,nb_gaussians in enumerate(list_nb_gaussians):
        print('nb gaussians', nb_gaussians)
        print('vérité terrain')
        _, coeffs, _, ssim_gt, opt_nb_gauss_gt = heuristic_gmm_fitting(image, size, np.zeros((3,3)), sigma, 10**-4, 0.1, nb_gaussians)
        ssims_gt[i] = ssim_gt
        nb_zeros_coeffs = len(coeffs[coeffs < 0.2])
        list_nb_zeros_coeffs.append(nb_zeros_coeffs)
        """
        opt_nb_gauss_views = np.zeros(len(views))
        for v in range(len(views)):
            print(f'vue {v}')
            _, coeffs, _, ssim_v, opt_nb_gauss = heuristic_gmm_fitting(views[v], size, cov_PSF, sigma, 10 ** -4,
                                                                               0.1, nb_gaussians)
            ssims_views[i, v] = ssim_v
            opt_nb_gauss_views[v] = opt_nb_gauss
        """

    """
    plt.plot(list_nb_gaussians, list_nb_zeros_coeffs)
    plt.vlines(opt_nb_gauss, 0, np.max(list_nb_zeros_coeffs),label=color='red')
    plt.xlabel('nombre de gaussiennes')
    plt.ylabel('nombre de coefficients réduits à 0 (inférieurs à 0.01)')
    plt.title('nombre de gaussiennes optimal pour représenter une vue')
    plt.grid()
    plt.show()
    """
    colors = gen_colors(4)
    plt.plot(list_nb_gaussians, ssims_gt, marker='X')
    plt.xlabel('nombre de gaussiennes')
    plt.ylabel("ssim entre l'image et la mixture de gaussiennes")
    ma = min(np.max(ssims_gt)+0.1, 1)
    plt.vlines(opt_nb_gauss_gt, np.min(ssims_gt), ma, label='nombre de gaussiennes trouvé par la méthode des niveaux de gris', color='red')
    plt.title("Représentation d'une vue par une mixture de gaussiennes. Ssim en fonction du nombre de gaussiennes")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.title("Représentation d'une vue sous la forme d'une mixture de gaussiennes")
    """
    for v in range(3):
        plt.plot(list_nb_gaussians, ssims_views[:, v])
        plt.vlines(opt_nb_gauss_views[v], 0, np.max(ssims_views[:, v]), label=f'vue {v+1}', color=colors[v+1])

    plt.grid()
    plt.show()
    print(f'moyenne du nombre de gaussiennes idéal sur toute les vues {np.mean(opt_nb_gauss_views)}')
    print(f'nombre idéal de gaussiennes de la vérité terrain', opt_nb_gauss_gt)
    """


    """
    save('tests_on_particules/GMM_GT/original_GT.tif', image)
    save('tests_on_particules/GMM_GT/GM_GT_centriole.tif', GM)
    write_array_csv(centers, 'tests_on_particules/GMM_GT/centers.csv')
    write_array_csv(coeffs, 'tests_on_particules/GMM_GT/coeffs.csv')
    """