import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from metrics_and_visualisation.fourier_shell_correlation import calc_fsc, fsc2res, conical_fourrier_shell_correlation
from skimage.metrics import structural_similarity as ssim
from common_image_processing_methods.rotation_translation import *
from manage_matplotlib.graph_setup import set_up_graph
from matplotlib.ticker import ScalarFormatter


def fsc(im1, im2, side=None, cutoff=0.143, plot_path=None, pixel_size=None):
    if side is None:
        side = im1.shape[0]
    resn, x, y, resx = fsc2res(calc_fsc(im1, im2, side), cutoff=cutoff, return_plot=True)
    if plot_path is not None:
        plot_fsc_graph(x, y, plot_path, cutoff, pixel_size=pixel_size)
    #print('resn', resn)
    #print('resx', resx)
    #plt.plot(x, y)
    #plt.show()
    return resn, resx


def plot_fsc_graph(x,y, plot_path, cutoff, pixel_size=None):
    # pixel size in nm
    set_up_graph()
    indices = np.where(y<=cutoff)[0]
    if pixel_size is not None:
        x = x/pixel_size
    plt.plot(x, y, linewidth=4, zorder=1)
    min_val = np.min(y) - 0.02
    index = indices[0] if len(indices) > 0 else -1
    x_cutoff = x[index]
    y_cutoff = y[index]
    ligne_pointille_hor = x[0:index]
    #plt.tick_params(axis='x', labelright=True, labelleft=False, which='both', labelsize=70)

    # Ajouter "* 10**-3" à droite de l'axe des abscisses
    plt.plot(ligne_pointille_hor, [y_cutoff]*len(ligne_pointille_hor), linestyle="dashed", color="red", linewidth=4)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlim((0, x[-1]))
    plt.ylim((min_val, 1.05))
    if pixel_size is None:
        plt.xlabel("Fréquence (1/pixel)")
    else:
        plt.xlabel('Fréquence (1/nm)')

    plt.ylabel("Coefficient de corrélation")


    plt.vlines(x_cutoff, min_val, y_cutoff, linestyle="dashed", color="red", linewidth=4)
    plt.scatter(x_cutoff, y_cutoff, color='red', marker="X", s=700, zorder=2)


    plt.annotate(f'{round(10**2*x_cutoff, 2)}', (x_cutoff, 0), textcoords="offset points",
                 xytext=(-65, -12), ha='center', color='red', fontsize=60)

    plt.annotate(f'{cutoff}', (0, y_cutoff), textcoords="offset points",
                 xytext=(90, -50), ha='center', color='red', fontsize=60)

    plt.grid()
    plt.savefig(plot_path)
    plt.close()


def psnr(im1, im2):
    return -np.log(np.mean((im1-im2)**2))


def f1_score(X, Y):
    return 2*np.sum(X*Y)/(np.sum(X)+np.sum(Y))


def f1_score_sqrt(X, Y):
    return 2 * np.sum(X * Y) / (np.sum(X) + np.sum(Y))

"""
def f1_score(X, Y):
    I = np.sum(X * Y)
    U = np.sum(X + Y)
    return (2*I)/(U+I)
"""


def jaccard_index(X, Y):
    I = np.sum(X * Y)
    U = np.sum(X + Y)
    return I/U


def normalized_correlation(im1, im2):
    return np.sum(im1*im2)/np.sqrt(np.sum(im1**2)*np.sum(im2**2))


def ssim_after_thresholding(im1, im2, thresh=0.5):
    im1[im1<thresh] = 0
    im2[im2<thresh] = 0
    im1[im1>thresh] = 1
    im2[im2>thresh] = 1
    return ssim(im1, im2)


def ssim_after_thresholding_2(im1, im2, thresh=0.5):
    im1[im1 < thresh] = 0
    im1[im1 > thresh] = 1
    return ssim(im1, im2)


def dice_after_thresholding(im1, im2, thresh=0.6):
    im1[im1<thresh] = 0
    im2[im2<thresh] = 0
    im1[im1>thresh] = 1
    im2[im2>thresh] = 1
    return dice(im1, im2)


def mutual_info_score_for_nd_arrays(im1, im2):
    im1_flatten = im1.flatten()
    im2_flatten = im2.flatten()
    im1_flatten = np.uint(255*im1_flatten)
    im2_flatten = np.uint(255*im2_flatten)
    return mutual_info_score(im1_flatten, im2_flatten)


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def measure_of_isotropy(im1, im2, cutoff=0.2, nb_radius=49, nb_sectors=20, sigma_radius=7, coeff_kernel_axes=5):
    thetas, phis, _ = discretize_sphere_uniformly(nb_sectors)
    x, y, z = conversion_2_first_eulers_angles_cartesian(np.array(thetas), np.array(phis))
    axes = np.array([x, y, z])
    conical_fsc, radiuses_frequencies = conical_fourrier_shell_correlation(im1, im2, axes, nb_radius,
                                                                           coeff_kernel_axes, sigma_radius)
    ress_n_axis = np.zeros(nb_sectors)
    for i in range(nb_sectors):
        fsc = np.vstack((radiuses_frequencies, conical_fsc[:, i])).T
        _, res = fsc2res(fsc, cutoff=cutoff)
        ress_n_axis[i] = res
    ress_n_axis = ress_n_axis[ress_n_axis > 0]
    return np.std(1/ress_n_axis)
    # return np.exp(-np.std(1/ress_n_axis)/3)
    # return 1/(1+10*np.std(ress_n_axis))


def compute_4d_metric(registered_est_vol, gt, metric=fsc):
    avg_met_val = 0
    for t in range(len(registered_est_vol)):
        met_val = metric(registered_est_vol[t], gt[t])
        avg_met_val += met_val
    avg_met_val/= len(registered_est_vol)
    return avg_met_val


if __name__ == '__main__':
    from manage_files.paths import *
    from data_generation.generate_data import get_PSF_with_stds
    from manage_files.read_save_files import read_image
    from skimage import io
    from common_image_processing_methods.others import normalize

    im = read_image('/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/gmm_reconstruction/sig_0.03/'
           'nb_gaussians_2/test_5/recons.tif')
    gt = read_image(f'{PTH_GT}/recepteurs_AMPA.tif')
    print(ssim(im, gt))
    1/0


    sim_data = False
    fsc_view_gt = False
    sig_z = 5
    sig_xy = 1  # used only for the simulated gaussians
    cut_off = 0.5
    n_test=10
    PATH_RESULTS_HPC = "/home/eloy/Documents/archives/stage_reconstruction_spfluo/results_hpc"
    pth = '/home/eloy/Documents/article_reconstruction_micro_fluo/TCI23'
    im_fortun_recons = ['AMPA_5p_lambda1e-3', 'clathrine_5p_lambda1e-3', 'emd_0680_5p_lambda1e-3',
                      'HIV_5p_lambda1e-3']

    for i,part_name in enumerate(["recepteurs_AMPA", "clathrine", "emd_0680", "HIV-1-Vaccine_prep"]):
        avg_fsc = 0
        pth2 = f'{PTH_GT}/{part_name}.tif'
        im2 = io.imread(pth2)
        """
        pth_fortun = f'{pth}/{im_fortun_recons[i]}_registered.tif'
        pth2 = f'{PTH_GT}/{part_name}.tif'
        im2 = io.imread(pth2)
        im_fortun = io.imread(pth_fortun)
        fsc_fortun = fsc(im2, im_fortun, cutoff=cut_off)
        print(f'fsc fortun {part_name}', fsc_fortun)
        ssim_forun = ssim(im2, im_fortun)
        print('ssim fortun', ssim_forun)
        """
        avg_ssim = 0
        for t in range(n_test):
            pth_folder_res = f'{PATH_RESULTS_HPC}/test_gt/{part_name}/test_{t}'
            pth1 = f'{pth_folder_res}/recons_registered.tif'
            #pth1 = f'/home/eloy/Documents/stage_reconstruction_spfluo/results/{part_name}/known_angles_sig_z_5/test_1/recons.tif'
            #pth1 = f'/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/tomographic_reconstruction/recepteurs_AMPA/nb_views_10/vol{t+1}_registered.tif'
            """load two images from paths"""
            im1 = io.imread(pth1)
            #isotropy = measure_of_isotropy(im1, im2
                                           #, cutoff=cut_off)
            #print('isotropy ', isotropy)
            fsc_val = fsc(normalize(im1), normalize(im2), cutoff=cut_off)
            ssim_val = ssim(im1, im2)
            avg_ssim += ssim_val
            # print('fsc val', fsc_val)
            avg_fsc += fsc_val
        avg_fsc/=n_test
        avg_ssim/=n_test
        print(f'fsc val {part_name}', avg_fsc)
        print(f'ssim val {part_name}', avg_ssim)


