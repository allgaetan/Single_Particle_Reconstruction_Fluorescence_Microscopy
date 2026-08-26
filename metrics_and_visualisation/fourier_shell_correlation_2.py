import os

import matplotlib.pyplot as plt

from manage_matplotlib.graph_setup import *
from manage_matplotlib.plot_graph import plot_graphs
import numpy as np
import pylab as pl
from scipy import ndimage
from skimage import io
from common_image_processing_methods.rotation_translation import *
from common_image_processing_methods.others import *
from scipy.interpolate import griddata
from manage_files.read_save_files import *
from time import time

CUTOFF_DEFAULT_VAL = 0.2


def generate_image_with_ft(coeffs, frequencies, im_size, lenght):
    """coeffs : shape (nb_dim, len(frequencies))"""
    nb_dim = coeffs.shape[0]
    X = np.linspace(0, lenght, im_size)
    im_shape = tuple([im_size for _ in range(nb_dim)])
    ft_im = np.zeros(im_shape)
    for k in range(coeffs.shape[0]):
        lin_comb_sin = np.zeros(im_size)
        for i in range(coeffs.shape[1]):
            cos = coeffs[k][i] * np.cos(2 * np.pi * frequencies[k][i] * X)
            lin_comb_sin += cos

        squeeze_shape = np.ones(nb_dim)
        squeeze_shape[k] = im_size
        lin_comb_sin = lin_comb_sin.reshape(tuple(squeeze_shape.astype(np.uint8)))
        for j in range(nb_dim):
            if j != k:
                lin_comb_sin = np.repeat(lin_comb_sin, im_size, axis=j)
        ft_im += lin_comb_sin
    im = np.fft.ifftn(ft_im)
    return im, ft_im


def calc_fsc(rho1, rho2, side):
    """ Calculate the Fourier Shell Correlation between two electron density maps.
    side : half of the size of the image in Angstrom"""
    df = 1.0 / side
    n = rho1.shape[0]
    qx_ = np.fft.fftfreq(n) * n * df
    qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing='ij')
    qx_max = qx.max()
    qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    print('qr shape', qr.shape)
    print('qr', qr)
    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0])

    nbins = int(qmax / qstep)
    qbins = np.linspace(0, nbins * qstep, nbins + 1)
    print('q bine shape', qbins.shape)
    print('q bins', qbins)
    # create an array labeling each voxel according to which qbin it belongs

    qbin_labels = np.searchsorted(qbins, qr, "right")
    print('q bin labels shape', qbin_labels.shape)
    print('q bin labels', qbin_labels)
    qbin_labels -= 1
    F1 = np.fft.fftn(rho1)
    F2 = np.fft.fftn(rho2)
    numerator = ndimage.sum(np.real(F1 * np.conj(F2)), labels=qbin_labels,
                            index=np.arange(0, qbin_labels.max() + 1))
    term1 = ndimage.sum(np.abs(F1) ** 2, labels=qbin_labels,
                        index=np.arange(0, qbin_labels.max() + 1))
    term2 = ndimage.sum(np.abs(F2) ** 2, labels=qbin_labels,
                        index=np.arange(0, qbin_labels.max() + 1))
    denominator = (term1 * term2) ** 0.5
    FSC = numerator / denominator
    qidx = np.where(qbins < qx_max)
    return np.vstack((qbins[qidx], FSC[qidx])).T


def fsc2res(fsc, cutoff=CUTOFF_DEFAULT_VAL, return_plot=False):
    """Calculate resolution from the FSC curve using the cutoff given.
    fsc - an Nx2 array, where the first column is the x axis given as
          as 1/resolution (angstrom).
    cutoff - the fsc value at which to estimate resolution, default=0.5.
    return_plot - return additional arrays for plotting (x, y, resx)
    """
    x = np.linspace(fsc[0, 0], fsc[-1, 0], 1000)
    y = np.interp(x, fsc[:, 0], fsc[:, 1])
    if np.min(fsc[:, 1]) > cutoff:
        # if the fsc curve never falls below zero, then
        # set the resolution to be the maximum resolution
        # value sampled by the fsc curve
        resx = np.max(fsc[:, 0])
        # print("Resolution: < %.1f A (maximum possible)" % resn)
    elif np.max(fsc[:, 1]) < cutoff:
        resx = x[0]
    else:
        idx = np.where(y >= cutoff)
        # resi = np.argmin(y>=0.5)
        # resx = np.interp(0.5,[y[resi+1],y[resi]],[x[resi+1],x[resi]])
        resx = np.max(x[idx])
        # print("Resolution: %.1f A" % resn)
    resn = float(1. / (resx + 10 ** -3))
    if return_plot:
        return resn, x, y, resx
    else:
        return resn, resx


def compute_frc(
        image_1: np.ndarray,
        image_2: np.ndarray,
        bin_width: int = 2.0
):
    """ Computes the Fourier Ring/Shell Correlation of two 2-D images

    :param image_1:
    :param image_2:
    :param bin_width:
    :return:
    """
    image_1 = image_1 / np.sum(image_1)
    image_2 = image_2 / np.sum(image_2)
    f1, f2 = np.fft.fft2(image_1), np.fft.fft2(image_2)
    af1f2 = np.real(f1 * np.conj(f2))
    af1_2, af2_2 = np.abs(f1) ** 2, np.abs(f2) ** 2
    nx, ny = af1f2.shape
    x = np.arange(-np.floor(nx / 2.0), np.ceil(nx / 2.0))
    y = np.arange(-np.floor(ny / 2.0), np.ceil(ny / 2.0))
    distances = list()
    wf1f2 = list()
    wf1 = list()
    wf2 = list()
    for xi, yi in np.array(np.meshgrid(x, y)).T.reshape(-1, 2):
        distances.append(np.sqrt(xi ** 2 + xi ** 2))
        xi = int(xi)
        yi = int(yi)
        wf1f2.append(af1f2[xi, yi])
        wf1.append(af1_2[xi, yi])
        wf2.append(af2_2[xi, yi])

    bins = np.arange(0, np.sqrt((nx // 2) ** 2 + (ny // 2) ** 2), bin_width)
    f1f2_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1f2
    )
    f12_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1
    )
    f22_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf2
    )
    density = f1f2_r / np.sqrt(f12_r * f22_r)
    return density, bin_edges


def conical_fourier_shell_correlation_split_computation(im1, im2, axes, nb_radius, coeff_kernel_angles, sigma_radius,
                                                        side=None, set_size=30, transposition=(1, 2, 0)):
    """split computation of the conical fsc in several sets to ovoid memory errors"""
    nb_sectors = axes.shape[1]
    nb_sets = int(np.ceil(nb_sectors / set_size))
    conical_fsc = np.zeros((nb_radius, nb_sectors))
    for i in range(nb_sets):
        start_idx = i * set_size
        end_idx = min((i + 1) * set_size, nb_sectors)
        set_axes = axes[:, start_idx:end_idx]
        t = time()
        conical_fsc_set, radiuses_frequencies, kernel_angles_threshold = conical_fourrier_shell_correlation(im1, im2,
                                                                                                            set_axes,
                                                                                                            nb_radius,
                                                                                                            coeff_kernel_angles,
                                                                                                            sigma_radius,
                                                                                                            side=side,
                                                                                                            transposition=transposition)
        conical_fsc[:, start_idx:end_idx] = conical_fsc_set
    return conical_fsc, radiuses_frequencies, kernel_angles_threshold


def conical_fourrier_shell_correlation(im1, im2, axes, nb_radius, coeff_kernel_angles, sigma_radius, side=None,
                                       transposition=(1, 2, 0), set_size=30):
    """ computes the conical Fourier shell correlation between the two images im1 and im2.
    im1 and im2 are assumed to be cubic and to have the same shape
    side =  lenght of the volume
    Returns 2d array of size (nb_radius, nb_sectors), which contains the values of the fourier correlation between the two images
    (nb_sectors = axes.shape[1])
    associated to each raidus and each sector"""

    im1 = np.transpose(im1, transposition)
    im2 = np.transpose(im2, transposition)
    n = im1.shape[0]
    if side is None:
        side = n
    df = 1 / side
    sigma_radius /= side
    """creation of the frequency frequels"""
    qx_ = np.fft.fftfreq(n) * n * df
    frequels = np.array(np.meshgrid(qx_, qx_, qx_, indexing='ij'))
    frequels_reshaped = np.reshape(frequels, (3, n ** 3))

    """Kernel values associated to radius"""
    radiuses_grid = np.expand_dims(np.sqrt(np.sum(frequels_reshaped ** 2, axis=0)), 0)
    radius_max = np.max(frequels_reshaped)
    radiuses = np.linspace(0, radius_max, nb_radius)
    radiuses_grid_repeated = np.repeat(radiuses_grid, len(radiuses), axis=0)
    radiuses_repeated = np.repeat(np.expand_dims(radiuses, 1), radiuses_grid_repeated.shape[1], axis=1)
    kernel_radius = np.exp(-(radiuses_repeated - radiuses_grid_repeated) ** 2 / (2 * sigma_radius ** 2))
    # kernel_radius = 1 * (kernel_radius>=1/2*np.max(kernel_radius))
    # kernel_radius/=np.sum(kernel_radius)

    """Kernel values associated to angles"""

    normalization = np.repeat(np.expand_dims(np.sqrt(np.sum(axes ** 2, axis=0)), 0), 3, axis=0)
    axes = axes / normalization
    normalizations_vals = np.repeat(radiuses_grid, 3, axis=0)
    normalizations_vals[normalizations_vals == 0] = 1
    frequels_reshaped /= normalizations_vals
    kernel_angles = np.exp(-coeff_kernel_angles * np.arccos(np.abs(np.dot(axes.T, frequels_reshaped))))

    theta_frequels = []
    phi_frequels = []
    for i in range(frequels_reshaped.shape[1]):
        _, theta, phi = conversion_cartesian_polar(*frequels_reshaped[:, i], degree=True)
        if theta != None:
            theta_frequels.append(theta)
            phi_frequels.append(phi)
        else:
            theta_frequels.append(0)
            phi_frequels.append(0)
    plt.figure(figsize=(20, 10))
    for id in [0,1]:
        kernel_angles_exemple = 1 * (kernel_angles[id, :] >=1/2*np.max(kernel_angles))
        plot_map(kernel_angles_exemple, theta_frequels, phi_frequels, 0, 1)
        theta = thetas[id]
        if theta > 180:
            theta -= 360
        if theta <= -180:
            theta += 360
        plt.scatter(theta, phis[id], marker='X', c='r')
    plt.show()

    1/0


    print('shp angles', kernel_angles.shape)
    # kernel_angles = 1* (kernel_angles >= 3/4*np.max(kernel_angles))
    # kernel_angles/=np.sum(kernel_angles)
    """defines the global kernel by multiplying the two previous"""
    nb_sets = int(np.ceil(nb_sectors / set_size))
    ma = 0
    print(nb_sets)
    for i in range(nb_sets):
        start_idx = i * set_size
        end_idx = min((i + 1) * set_size, nb_sectors)
        kernel_angles_sub_set = kernel_angles[start_idx:end_idx, :]
        kernel_angles_repeated = np.repeat(np.expand_dims(kernel_angles_sub_set, axis=0), kernel_radius.shape[0], axis=0)
        kernel_radius_repeated = np.repeat(np.expand_dims(kernel_radius, axis=1), kernel_angles_sub_set.shape[0], axis=1)
        global_kernel = kernel_angles_repeated * kernel_radius_repeated
        ma_ker = np.max(global_kernel)
        if ma_ker > ma:
            ma = ma_ker
    print('la')
    # global_kernel = 1 * (global_kernel >= 1 / 2 * np.max(global_kernel))
    """compute the fourier transofrm of images and repeat to match kernel shape"""
    F1 = np.fft.fftn(im1)
    F2 = np.fft.fftn(im2)
    # F1 = im1
    # F2 = im2
    F1_flatten = np.reshape(F1, (1, 1, n * n * n))
    F2_flatten = np.reshape(F2, (1, 1, n * n * n))
    conical_fsc = np.zeros((nb_radius, nb_sectors))
    for i in range(nb_sets):
        start_idx = i * set_size
        end_idx = min((i + 1) * set_size, nb_sectors)
        kernel_angles_sub_set = kernel_angles[start_idx:end_idx, :]
        kernel_angles_repeated = np.repeat(np.expand_dims(kernel_angles_sub_set, axis=0), kernel_radius.shape[0],
                                           axis=0)
        kernel_radius_repeated = np.repeat(np.expand_dims(kernel_radius, axis=1), kernel_angles_sub_set.shape[0],
                                           axis=1)
        global_kernel = kernel_angles_repeated * kernel_radius_repeated
        global_kernel_thresh =  1* (global_kernel>=1/2*ma)
        F1_repeated = np.repeat(np.repeat(F1_flatten, global_kernel_thresh.shape[0], axis=0), global_kernel_thresh.shape[1], axis=1)
        F2_repeated = np.repeat(np.repeat(F2_flatten, global_kernel_thresh.shape[0], axis=0), global_kernel_thresh.shape[1], axis=1)
        print('shp', F1_repeated.shape)
        """computes the conical fourier shell correlation"""
        numerator = np.sum(global_kernel_thresh * np.real(F1_repeated * np.conj(F2_repeated)), axis=2)
        first_factor_denom = np.sum(global_kernel_thresh * np.real(F1_repeated * np.conj(F1_repeated)), axis=2)
        second_factor_denom = np.sum(global_kernel_thresh * np.real(F2_repeated * np.conj(F2_repeated)), axis=2)
        denominator = np.sqrt(first_factor_denom * second_factor_denom)
        conical_fsc_set = numerator / denominator
        conical_fsc[:, start_idx:end_idx] = conical_fsc_set

    return conical_fsc, radiuses


def plot_map(vals_to_plot, thetas, phis, vmin=None, vmax=None, init_fig=True):
    if init_fig:
        plt.figure(figsize=(20, 10))
    min_theta, max_theta, min_phi, max_phi = np.min(thetas), np.max(thetas), \
                                             np.min(phis), np.max(phis)
    # min_theta, max_theta, min_phi, max_phi = 0,360,0,180
    grid_theta, grid_phi = np.mgrid[min_theta:max_theta:500j,
                           min_phi:max_phi:500j]
    grid_z = griddata(np.array([thetas, phis]).T, vals_to_plot, (grid_theta, grid_phi), method='nearest')
    mean_non_nan = np.nanmean(grid_z)
    plt.imshow(np.nan_to_num(grid_z.T, nan=mean_non_nan), extent=(min_theta, max_theta, min_phi, max_phi),
               origin='lower', vmin=vmin, vmax=vmax)  # , origin='lower', extent=(0, 1, 0, 1))
    plt.colorbar()
    # plt.title("Resolution map (conical fsc computed between a reconstruction and ground truth image)")
    plt.xlabel("φ_1 (°)")
    plt.ylabel("φ_2 (°)")


def find_cutoffs_conical_fsc(radiuses_frequencies, conical_fsc, cutoff=CUTOFF_DEFAULT_VAL):
    radiuses_cut_off = np.zeros(conical_fsc.shape[1])
    for i in range(conical_fsc.shape[1]):
        if cutoff < np.min(conical_fsc[:, i]):
            res = radiuses_frequencies[-1]
        else:
            try:
                idx_cut_off = np.min(np.where(conical_fsc[:, i] <= cutoff))
                res = radiuses_frequencies[idx_cut_off]
            except:
                print('i', i)
                res = 0
                print('cFSC', conical_fsc[:, i])

        if res == 0:
            res = radiuses_frequencies[1]
        radiuses_cut_off[i] = res
    return radiuses_cut_off


def plot_resolution_map(radiuses_frequencies, thetas, phis, conical_fsc, cutoff=CUTOFF_DEFAULT_VAL, vmin=0, vmax=0.5):
    radiuses_cut_off = find_cutoffs_conical_fsc(radiuses_frequencies, conical_fsc, cutoff)
    plot_map(radiuses_cut_off, thetas, phis, vmin, vmax)
    return radiuses_cut_off


if __name__ == '__main__':
    from data_generation.generate_data import get_PSF_with_stds
    from manage_files.paths import *
    from manage_files.read_save_files import make_dir

    spartran = True
    known_angles = False
    conical_fsc_view_recons = []
    part_name = 'recepteurs_AMPA'
    transposition = (1, 2, 0)
    real_data_reconstructions = False
    for part_name in ["emd_0680", "recepteurs_AMPA", "HIV-1-Vaccine_prep", "clathrine", "Vcentriole_prep"]:
        for spartran in [False, True]:
            conical_fsc_view_recons = []
            for fsc_view_gt in [True, False]:
                sim_data = False
                use_given_axes = False
                sig_z = 10
                sig_xy = 2  # used only for the simulated gaussians
                if not real_data_reconstructions:
                    pth_folder_res = f'{PATH_RESULTS_HPC}/test_gt/{part_name}/test_1'
                    if fsc_view_gt:
                        pth1 = f'{PATH_VIEWS}/{part_name}/single_view/sig_z_{sig_z}/view_10.tif'
                    else:
                        if not spartran:
                            if not known_angles:
                                pth1 = f'{pth_folder_res}/recons_registered.tif'
                            else:
                                pth1 = f'{PTH_LOCAL_RESULTS}/{part_name}/knwon_angles/recons.tif'
                        else:
                            pth1 = f'{PATH_RESULTS_SCIPION}/tomographic_reconstruction/{part_name}/nb_views_80/vol1_registered.tif'
                    pth2 = f'{pth_folder_res}/ground_truth.tif'
                else:
                    pth1 = f'{PATH_REAL_DATA}/picked_centrioles_preprocessed_results_0/recons.tif'
                    pth2 = f'{PATH_REAL_DATA}/picked_centrioles_preprocessed_results_1/recons_shift_registered.tif'
                sig_noise = 1
                if sim_data:
                    """generates 2 images from a gaussian by adding 2 differnet realisations of a white noise"""
                    f1 = 2 * np.arange(1, 4)
                    f2 = f1
                    frequencies_1 = np.array([f2, f2, f1])
                    frequencies_2 = np.array([f2, f2, f1])
                    coeffs_z_1 = np.random.random(len(f1))
                    coeffs_z_2 = np.random.random(len(f1))
                    # coeffs_z = [0,0,0,0,0,0,0,0,0,1]
                    # coeffs_x = [0,1,0,1,0,1,0,1,0,1]
                    # coeffs_y = [1,0,1,0,1,0,1,0,1,0]
                    coeffs_x_1 = np.random.random(len(f1))
                    coeffs_x_2 = np.random.random((len(f1)))
                    coeffs_y_1 = np.random.random(len(f1))
                    coeffs_y_2 = np.random.random(len(f1))

                    zeros = np.zeros(len(f1))
                    coeffs_1 = np.array([coeffs_z_1,
                                         coeffs_x_1,
                                         coeffs_y_1])

                    coeffs_2 = np.array([coeffs_z_2, coeffs_x_2, coeffs_y_2])
                    # coeffs_1 = [[]]
                    # coeffs_1 = np.random.random((3, len(frequencies_1)))
                    # coeffs_2 = np.random.random((3, len(frequencies_2)))
                    sz = 50
                    _, im1 = generate_image_with_ft(coeffs_1, frequencies_1, sz, 1) + np.random.normal(0, sig_noise,
                                                                                                       (sz, sz, sz))
                    _, im2 = generate_image_with_ft(coeffs_2, frequencies_2, sz, 1) + np.random.normal(0, sig_noise,
                                                                                                       (sz, sz, sz))
                    save('im_sin.tif', im1)
                    sphere = get_PSF_with_stds(50, sig_z, sig_xy)
                    # im1 = sphere + sig_noise * np.random.random(sphere.shape)
                    # im2 = sphere + sig_noise * np.random.random(sphere.shape)
                else:
                    from common_image_processing_methods.others import resize

                    """load two images from paths"""
                    sz = 50
                    im1 = resize(io.imread(pth1), (sz, sz, sz))
                    im2 = resize(io.imread(pth2), (sz, sz, sz))
                    # im1 += np.random.normal(0, sig_noise, im1.shape)
                    # im2 += np.random.normal(0, sig_noise, im2.shape)
                    # size = (10,10,10)
                    # im1 = window_fft(im1, size, sig_noise)
                    # im2 = window_fft(im2, size, sig_noise)

                nb_sectors = 2
                nb_radius = 30
                sig_radus = 7
                coeff_kernel_axes = 5
                """defines the orientations along which to compute the fsc"""
                if use_given_axes:
                    axes = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
                    labels = ['z axis', 'x axis',
                              'y axis']  # defines the labels associated to the axes to put in legend of the graph
                else:
                    # points_to_add = [(0,0), (0,180), (360,0), (360,180), (0,90), (360,90), (180,180), (155,142), (241,152), (180,0), (156,158),(314,24)]
                    # points_to_add = [(0, 0), (0, 180), (360, 0), (360, 180), (0, 90), (360, 90), (180, 180), (180, 0),
                    #                 (180, 90)]

                    points_to_add = [(0, 0)]
                    thetas, phis, _ = discretize_sphere_uniformly(nb_sectors - len(points_to_add))
                    thetas = list(thetas)
                    phis = list(phis)
                    print('list thetas', thetas)
                    print('list phis', phis)
                    for i in range(len(points_to_add)):
                        thetas.append(points_to_add[i][0])
                        phis.append(points_to_add[i][1])
                    x, y, z = conversion_2_first_eulers_angles_cartesian(np.array(thetas), np.array(phis))
                    axes = np.array([x, y, z])
                    labels = [None] * nb_sectors

                conical_fsc, radiuses_frequencies = conical_fourrier_shell_correlation(
                    im1, im2, axes, nb_radius, coeff_kernel_axes, sig_radus,
                    transposition=transposition)


                conical_fsc_view_recons.append(conical_fsc)
                print('conical fsc shape', conical_fsc.shape)
                # fsc_isotrop = calc_fsc(im1, im2, im1.shape[0])
                # plot_graphs(fsc_isotrop[:, 0], [fsc_isotrop[:, 1]], 'frequencies (1/pixel)', 'correlation coefficient', '')

                """
                for i in range(len(conical_fsc[1])):
                    if use_given_axes:
                        plt.plot(radiuses_frequencies, conical_fsc[:, i], label=labels[i])
                    else:
                        plt.plot(radiuses_frequencies, conical_fsc[:, i])

                plt.xlabel('frequencies (1/pixel)')
                plt.ylabel('correlation coefficient')
                """
                if not real_data_reconstructions:
                    if not sim_data:
                        title = f'fsc curves between {part_name} and a view, sig_z={sig_z}' if fsc_view_gt else \
                            f'fscs curves between {part_name} and a reconstruction, sig_z={sig_z}'
                        save_path = f'{PTH_ILLUSTRATIONS}/conical_fsc_3/{part_name}'
                        if fsc_view_gt:
                            save_name = f'cFSC_map_view_gt.png'
                        else:
                            if spartran:
                                save_path = f'{save_path}/spartran'
                            else:
                                save_path = f'{save_path}/recons'
                            save_name = 'cFSC_map.png'
                    else:
                        # title = f'Conical fsc curves on gaussians, sig_z = {sig_z}, sig_xy = {sig_xy}'
                        title = 'simulated images'
                        save_path = f'{PTH_ILLUSTRATIONS}/conical_fsc/simulated_data'
                        # save_name = f'gaussians_sig_z_{sig_z}_sig_xy_{sig_xy}'
                        save_name = 'simulated_images'
                    if known_angles:
                        save_name += 'known_angles'
                else:
                    save_path = f'{PTH_ILLUSTRATIONS}/conical_fsc/real_data_reconstructions'
                    save_name = 'assembly'
                    title = ''
                if use_given_axes:
                    save_name += '_given_axis'
                plot_graphs(radiuses_frequencies, conical_fsc.T, 'frequencies (1/pixel)', 'correlation coefficient',
                            title, labels, save_path, 'cfsc_graphs.png')
                if not use_given_axes:
                    plot_resolution_map(radiuses_frequencies, thetas, phis, conical_fsc, vmin=0, vmax=0.5)
                    save_figure(save_path, f'{save_name}.png')
                    plt.close()
                make_dir_and_write_array(conical_fsc, f'{save_path}/conical_fsc_array', f'{save_name}.csv')

            if len(conical_fsc_view_recons) == 2:
                cut_offs_recons = find_cutoffs_conical_fsc(radiuses_frequencies, conical_fsc_view_recons[1])
                cut_offs_view = find_cutoffs_conical_fsc(radiuses_frequencies, conical_fsc_view_recons[0])
                measure_of_isotropy_recons = np.std(cut_offs_recons)
                measure_of_isotropy_view = np.std(cut_offs_view)
                make_dir_and_write_array(np.array([[measure_of_isotropy_recons]]), f'{save_path}/measure_of_isotropy',
                                         f'{save_name}_recons.csv')
                make_dir_and_write_array(np.array([[measure_of_isotropy_recons]]),
                                         f'{save_path}/measure_of_isotropy', f'{save_name}_view.csv')
                plt.hist(cut_offs_recons)
                plt.title('hist cut off recons')
                save_figure(f'{save_path}/hisogramms', f'{save_name}_recons.png')
                plt.close()
                plt.hist(cut_offs_view)
                plt.title('hist cutt off view')
                save_figure(f'{save_path}/hisogramms', f'{save_name}_view.png')
                plt.close()
                gain = cut_offs_recons / cut_offs_view
                # gain[gain>5] = 5
                # print('cutoffs1', find_cutoffs_conical_fsc(radiuses_frequencies, conical_fsc_view_recons[1]))
                # print('cutoffs2', find_cutoffs_conical_fsc(radiuses_frequencies, conical_fsc_view_recons[0]))

                vmax = 8 if part_name == 'recepteurs_AMPA' else 3
                plot_map(gain, thetas, phis, vmax=vmax, vmin=0.2)
                save_figure(save_path, 'gain_map.png')
                plt.close()