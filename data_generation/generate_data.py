import shutil

import numpy as np

from manage_files.read_save_files import *
from common_image_processing_methods.others import normalize
from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly, rotation, translation
from scipy.ndimage import gaussian_filter, convolve
import os
from common_image_processing_methods.rotation_translation import get_rotation_matrix
from data_generation.simulate_partial_labelling import simulate_partial_labelling
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import make_grid, nd_gaussian, gaussian_mixture_isotrop_identical_gaussians

from numpy.fft import fftn, ifftn, ifftshift
from skimage.morphology import dilation
from create_centriole import simu_cylinder_rev, get_radius



def dilate_isotrop(im, kernel_sz):
    nb_dim = len(im.shape)
    kernel = np.ones(tuple([kernel_sz]*nb_dim))
    im = dilation(im, kernel)
    return im


def dilate_one_dimension(im, kernel_sz):
    nb_dim = len(im.shape)
    kernel = np.zeros(tuple([kernel_sz]*nb_dim))
    kernel[:, :, kernel_sz//2] = 1
    im = dilation(im, kernel)
    return im


def dilate_isotrop_plus_one_dim(im, dil_vals):
    return dilate_one_dimension(dilate_isotrop(im, dil_vals[0]), dil_vals[1])


def generate_and_save_data(folder_views, folder_ground_truth, ground_truth_name, params_data_gen, save_name_opt=1):
    delete_dir(folder_views)
    make_dir(folder_views)
    ground_truth = io.imread(f'{folder_ground_truth}/{ground_truth_name}')
    # ground_truth = center_barycenter(ground_truth)
    # ground_truth[ground_truth<0] = 0
    # save(f'{folder_ground_truth}/{ground_truth_name}', ground_truth)
    ground_truth = resize(ground_truth, tuple([params_data_gen["size"] for _ in range(params_data_gen["nb_dim"])]))
    views, rot_vecs, trans_vecs, _, _, roatated_convolved, rotated_images = generate_data(ground_truth, params_data_gen)
    for v in range(params_data_gen["nb_views"]):
        t = trans_vecs[v]
        rot_vec = rot_vecs[v]
        view = views[v]
        rot_im = rotated_images[v]
        rot_conv = roatated_convolved[v]
        if params_data_gen["nb_dim"] == 2:
            save_loc = f'{folder_views}/view_r_{rot_vec}_t_{t[0]}_{t[1]}_.png'
            save_loc_rot = f'{folder_views}/rot_views/view_r_{rot_vec}_t_{t[0]}_{t[1]}_.png'
            save_loc_conv = f'{folder_views}/rot_conv_views/view_r_{rot_vec}_t_{t[0]}_{t[1]}_.png'
            make_dir(f'{folder_views}/rot_views')
            make_dir(f'{folder_views}/rot_conv_views')
            save(f'{folder_views}/gt.png', ground_truth)
            save(save_loc_rot, rot_im)
            save(save_loc_conv, rot_conv)
        else:
            if save_name_opt == 1:
                save_loc = f'{folder_views}/view_r_{rot_vec[0]}_{rot_vec[1]}_{rot_vec[2]}_t_{t[0]}_{t[1]}_{t[2]}_.tif'
            else:
                save_loc = f'{folder_views}/view_100_{rot_vec[0]}_{rot_vec[1]}_{rot_vec[2]}_.tif'
        save(save_loc, view)
    print(f'views saved at location : {folder_views}')
    # save(f'{folder_views}/PSF.tif', PSF)
    return ground_truth


def convolve_noise(ground_truth, sig_z, sig_xy, snr=None):
    sigmas_PSF = np.zeros(3)
    sigmas_PSF[0] = sig_z
    for i in range(1, 3):
        sigmas_PSF[i] = sig_xy
    size = ground_truth.shape[0]
    PSF = get_PSF_with_stds(size, sig_z, sig_xy)
    filtered = gaussian_filter(ground_truth, sigmas_PSF)
    if snr is not None:
        power_filtered = np.mean(filtered ** 2)
        std_noise = np.sqrt(power_filtered / snr)
        noise = np.random.normal(0, std_noise, filtered.shape)
        filtered = filtered + noise
    return PSF, filtered


def get_PSF_with_stds(size, sig_z, sig_xy, nb_dim=3):
    grid_step = 2 / (size - 1)
    cov_PSF = grid_step ** 2 * np.eye(nb_dim)
    cov_PSF[0, 0] *= sig_z ** 2
    for i in range(1, nb_dim):
        cov_PSF[i, i] *= sig_xy ** 2
    grid = make_grid(size, nb_dim)
    PSF = nd_gaussian(grid, np.zeros(nb_dim), cov_PSF, nb_dim)
    PSF /= np.sum(PSF)
    return PSF


def convolve_fourier(im, psf):
    return ifftn(fftn(ifftshift(psf)) * fftn(im))


def generate_one_view(ground_truth, rot_vec, trans_vec, nb_dim, params_data_gen):
    sigmas_PSF = np.zeros(nb_dim)
    sigmas_PSF[0] = params_data_gen["sig_z"]
    for i in range(1, nb_dim):
        sigmas_PSF[i] = params_data_gen["sig_xy"]
    rot_mat = get_rotation_matrix(rot_vec, params_data_gen["convention"])
    rot_image, _ = rotation(ground_truth, rot_mat, order=params_data_gen["order"])
    rotated_translated = translation(rot_image, trans_vec)
    if not params_data_gen["no_psf"]:
        if params_data_gen["psf"] is None:
            filtered = gaussian_filter(rotated_translated, sigmas_PSF)
        else:
            filtered = convolve_fourier(rotated_translated, params_data_gen["psf"])
    else:
        filtered = rotated_translated
    if params_data_gen["projection"]:
        filtered = np.sum(filtered, axis=0)
    power_filtered = np.mean(filtered ** 2)
    std_noise = np.sqrt(power_filtered / params_data_gen["snr"])
    noise = np.random.normal(0, std_noise, filtered.shape)
    # noise = np.random.poisson(size=filtered.shape)
    filtered_noise = filtered + noise
    if params_data_gen["partial_labelling"]:
        filtered_noise = simulate_partial_labelling(filtered_noise, **params_data_gen["partial_labelling_args"])
    # filtered_noise[filtered_noise < 0] = 0
    return filtered_noise, rot_mat, rot_image, filtered


def generate_data(ground_truth, params_data_gen):
    thetas, phis, psis = discretize_sphere_uniformly(10000,360)
    nb_views = params_data_gen["nb_views"]
    nb_dim = params_data_gen["nb_dim"]
    if params_data_gen["rot_vecs"] is None:
        rot_vecs = np.zeros((nb_views, nb_dim))
    else:
        rot_vecs = params_data_gen["rot_vecs"]
    trans_vecs = np.zeros((nb_views, nb_dim))
    views = []
    dilatation_vals = []
    rot_mats = []
    rotated_images = []
    rotated_convolved = []
    for v in range(nb_views):
        if params_data_gen["rot_vecs"] is None:
            rotation_max = params_data_gen["rotation_max"]
            if nb_dim == 2:
                rot_vec = np.random.randint(0, 180)
            else:
                if rotation_max is None:
                    idx_viewing_direction, idx_in_plane_rot = np.random.randint(0, len(thetas)), np.random.randint(0,
                                                                                                                   len(psis))
                    theta, phi, psi = thetas[idx_viewing_direction], phis[idx_viewing_direction], psis[idx_in_plane_rot]

                else:
                    theta, phi, psi = np.random.random() * rotation_max[0], \
                                      np.random.random() * rotation_max[1], \
                                      np.random.random() * rotation_max[2]
                rot_vec = [round(theta, 1), round(phi, 1), round(psi, 1)]
            rot_vecs[v] = rot_vec
        else:
            rot_vec = rot_vecs[v]
        t = np.random.normal([0 for _ in range(nb_dim)], params_data_gen["sigma_trans_ker"])
        t = [round((t[i])) for i in range(nb_dim)]
        max_dil_val = params_data_gen["max_dil_val"]
        if max_dil_val is not None:
            dil_vals_view = []
            for i in range(len(max_dil_val)):
                print('dd', max_dil_val[i]+1)
                dil_val = np.random.randint(1, max_dil_val[i]+1)
                dil_vals_view.append(dil_val)
            if len(dil_vals_view) == 1:
                dil_vals_view = dil_vals_view[0]
            dilatation_vals.append(dil_vals_view)
            ground_truth_dilated = params_data_gen["dilatation_function"](ground_truth, dil_vals_view)
            # ground_truth_dilated *= 6134 / np.sum(ground_truth_dilated) (si l'on veut faire en sorte que la somme des pixels soit constante d'uen vue à l'autre)
        else:
            ground_truth_dilated = ground_truth
            dilatation_vals.append(1)

        view, rot_mat, rot_image, filtered = generate_one_view(ground_truth_dilated, rot_vec, t, nb_dim, params_data_gen)
        rotated_images.append(rot_image)
        rotated_convolved.append(filtered)
        views.append(view)
        trans_vecs[v] = t
        rot_mats.append(rot_mat)
    rot_vecs = np.array(rot_vecs)
    return views, rot_vecs, trans_vecs, rot_mats, dilatation_vals, rotated_convolved, rotated_images


def read_views(fold_name, nb_dim):
    """desires : int : les images sont redimensionné en un cube de taille (size, size, size)"""
    rot_vecs = []
    trans_vecs = []
    views, file_names = read_images_in_folder(fold_name)
    for i,view in enumerate(views):
        fn = file_names[i]
        #view = resize(view, tuple(desired_size for _ in range(nb_dim)))
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


    views = np.array(views)
    rot_vecs = np.array(rot_vecs)
    trans_vecs = np.array(trans_vecs) #.astype(np.float32)
    if len(views) == 0:
        print(f'WARNING : no view was read in {fold_name}, check the name of the folder')
    return views, rot_vecs, trans_vecs, file_names


def generate_rot_vec_trans_vec(param_data_gen, uniform_sphere_disc, zero_rotation=False):
    thetas, phis, psis = uniform_sphere_disc
    rot_max = param_data_gen["rotation_max"]
    if not zero_rotation:
        if rot_max is None:
            idx_viewing_direction, idx_in_plane_rot = np.random.randint(0, len(thetas)), np.random.randint(0, len(psis))
            theta, phi, psi = thetas[idx_viewing_direction], phis[idx_viewing_direction], psis[idx_in_plane_rot]

        else:
            theta, phi, psi = np.random.random() * rot_max[0], \
                              np.random.random() * rot_max[1], \
                              np.random.random() * rot_max[2]
        rot_vec = [round(theta, 1), round(phi, 1), round(psi, 1)]
    else:
        rot_vec = np.zeros(3)
    t = np.random.normal([0 for _ in range(param_data_gen["nb_dim"])], param_data_gen["sigma_trans_ker"])
    t = [round((t[i])) for i in range(param_data_gen["nb_dim"])]
    return rot_vec, t






def heterogene_views_from_centriole(save_fold, param_data_gen, nb_channels, het_vals_all_channels, cs=[103.253], zero_rotation=False, nb_missing_triplets=0, sig_gauss=0.03, homogene=False):
    """- save_fold  → path where views are saved
- params_data_gen → the dictionnary imported from default_params.py file. Eventually modify some values of this dictionnary
- nb_channels → nombre de canaux
- het_vals_all_channel → two dimensional array (or list of list) of shape (nb_channels, nb_views) het_avl_all_channels[c, l] corresponds to the heterogeneity values (i.e. the lenght of the generated centriole in the case of this function) of the l ^ th view of channel c
- cs → list of lenght nb_channels: cs[c] is the radius of the centriole of channel c (identical for all the views in the case of this function)
- zero_rotation: if True, all views are generating without rotating
- nb_missing triplets: number of triplets of microtubule romoved to the model
- sig_gauss: standard deviation of gaussian used to evaluate the point cloud on a grid
- homogene: if true generate homogene data and do not take into account the parameter het_vals_all_channelsl"""
    #make_dir(f'{save_fold}/gt_dilated')
    if save_fold is not None:
        if os.path.exists(save_fold):
            shutil.rmtree(save_fold)
    grid = make_grid(param_data_gen["size"], 3)
    uniform_sphere_disc = discretize_sphere_uniformly(10000, 360)
    gt_4d = [[] for c in range(nb_channels)]
    views = [[] for c in range(nb_channels)]
    rot_mats = []
    trans_vecs = []
    nb_points = 3000
    rot_vecs = []
    for v in range(param_data_gen["nb_views"]):
        # generation of poses, common for all channels
        rot_vec, t = generate_rot_vec_trans_vec(param_data_gen, uniform_sphere_disc, zero_rotation)
        rot_vecs.append(rot_vec)
        rot_mat = get_rotation_matrix(rot_vec, param_data_gen["convention"])
        rot_mats.append(rot_mat)
        trans_vecs.append(t)

        for c in range(nb_channels):
            if save_fold is not None:
                save_fold_channel = f'{save_fold}/c{c + 1}'
                make_dir(f'{save_fold}/c{c + 1}/gt')
            l = het_vals_all_channels[c][v]
            print('l', l)
            radius = get_radius(np.arange(1,14), -0.5, 5.4, cs[c])
            pc = simu_cylinder_rev(radius, 15, l, 15, nb_points, 110, nb_missing_triplets)
            # pc = normalize(pc, min=-0.7, max=0.7)
            pc = pc/200
            pc[:, 2] = pc[:, 2] - np.max(pc[:, 2])/2

            im = gaussian_mixture_isotrop_identical_gaussians(grid, [1]*len(pc), pc, sig_gauss, 3, 3)
            im = normalize(im, min=0, max=1)
            gt_4d[c].append(im)
            filtered, rot_mat, _, _ = generate_one_view(im, rot_vec, t, 3, param_data_gen)
            views[c].append(filtered)
            if save_fold is not None:
                save(f'{save_fold_channel}/view_{round(het_vals_all_channels[0][v], 2)}_{rot_vec[0]}_{rot_vec[1]}_{rot_vec[2]}_.tif', filtered)
        # save(f'{save_fold_channel}/gt_dilated/view_{round(het_vals_all_channels[0][v], 2)}_.tif', im)

    if save_fold is not None:
        for c in range(nb_channels):
            if not homogene:
                save_4d_for_chimera(gt_4d[c], f'{save_fold}/c{c+1}/gt/4d_vol_channel.tiff')
            else:
                save(f'{save_fold}/c{c+1}/gt/4d_vol_channel.tiff', gt_4d[c][0])

    views = np.transpose(np.array(views), (1,0,2,3,4))
    rot_mats = np.array(rot_mats)
    rot_vecs = np.array(rot_vecs)
    trans_vecs = np.array(trans_vecs)
    return views, het_vals_all_channels, rot_mats, rot_vecs, trans_vecs, gt_4d


def heterogene_views_from_centriole_2_degrees_of_freedom(save_fold, param_data_gen, het_vals_lenght,
                                                         het_vals_radius, sig_gauss=0.03):
    """- het_val_lenght → two dimensional array of shape (nb_channels, nb_views),  het_val_lenghts[c, l] corresponds to the lenght of the centrioke view l of channel c
- het_val_radius → two dimensional array of shape (nb_channels, nb_views),  het_val_radius[c, l] corresponds to the raidus of the centrioke view l of channel c"""
    if save_fold is not None:
        if os.path.exists(save_fold):
            shutil.rmtree(save_fold)
    grid = make_grid(param_data_gen["size"], 3)
    uniform_sphere_disc = discretize_sphere_uniformly(10000, 360)
    gt_4d = []
    views = []
    rot_mats = []
    trans_vecs = []
    nb_points = 3000
    rot_vecs = []
    for j in range(len(het_vals_lenght)):
        for k in range(len(het_vals_radius)):
            rot_vec, t = generate_rot_vec_trans_vec(param_data_gen, uniform_sphere_disc, False)
            rot_vecs.append(rot_vec)
            rot_mat = get_rotation_matrix(rot_vec, param_data_gen["convention"])
            rot_mats.append(rot_mat)
            trans_vecs.append(t)
            lenght = round(het_vals_lenght[j],2)
            radius = round(het_vals_radius[k],2)
            print('l', lenght)
            print('r', radius)
            radiuses = get_radius(np.arange(1, 14), -0.5, 5.4, radius)
            pc = simu_cylinder_rev(radiuses, 15, lenght, 15, nb_points, 110, 0)
            # pc = normalize(pc, min=-0.7, max=0.7)
            pc = pc / 200
            pc[:, 2] = pc[:, 2] - np.max(pc[:, 2]) / 2

            im = gaussian_mixture_isotrop_identical_gaussians(grid, [1] * len(pc), pc, sig_gauss, 3, 3)
            im = normalize(im, min=0, max=1)
            gt_4d.append(im)
            filtered, rot_mat, _, _ = generate_one_view(im, rot_vec, t, 3, param_data_gen)
            views.append(filtered)
            if save_fold is not None:
                make_dir(f'{save_fold}/c1')
                save(
                    f'{save_fold}/c1/view_{lenght}_{radius}_{rot_vec[0]}_{rot_vec[1]}_{rot_vec[2]}_.tif',
                    filtered)
                
    if save_fold is not None:
        make_dir(f'{save_fold}/c1/gt')
        save_4d_for_chimera(gt_4d, f'{save_fold}/c1/gt/4d_vol_channel.tiff')

if __name__ == '__main__':
    from reconstruction_with_dl.test_params.default_params import params_data_gen
    from common_image_processing_methods.others import *
    from manage_files.paths import PATH_PROJECT_FOLDER
    params_data_gen["size"] = 45
    params_data_gen["anis"] = 3
    params_data_gen["alpha"] = 1
    params_data_gen["sigma_trans_ker"] = 0
    sig_gauss = 0.03
    params_data_gen["nb_views"] = 1

    for snr in [10**-4,5*10**-4,10**-3,5*10**-3,10**-2,5*10**-2]:
        params_data_gen["snr"] = snr
        save_fold = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/views_snr_{snr}'
        #save_fold = f"/home/eloy/Documents/documents latex/these/images/fluofire/test_noise/snr_{snr_val}"
        # list_l_channel_1 = np.linspace(30, 350, 250)
        list_l_channel_1 = np.linspace(200, 200, 5)
        list_l_channel_2 = np.linspace(50,50, 5)
        # list_l_channel_1 = np.linspace(min_l[0], max_l[0], nb_views)
        # list_l_channel_2 = np.linspace(min_l[1], max_l[1], nb_views)
        cs = [103.25, 50]
        heterogene_views_from_centriole(save_fold=save_fold,param_data_gen=params_data_gen, nb_channels=1,
                                        het_vals_all_channels = [list_l_channel_1, list_l_channel_2], cs=cs, zero_rotation=True, homogene=True)


    #params_data_gen["nb_views"] = 250

    save_fold = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/multi_dim_het_missing_triplets_no_het'
    het_vals_lenght = np.linspace(30,350,50)
    het_vals_radius = np.linspace(50,110,30)
    heterogene_views_from_centriole_2_degrees_of_freedom(save_fold, params_data_gen, het_vals_lenght, het_vals_radius)
    1/0

     # discrete_states=[100,350,700])

    1/0

    folder = f'{PATH_VIEWS}/recepteurs_AMPA/noise'
    make_dir(folder)
    gt = read_image(f'{PTH_GT}/recepteurs_AMPA.tif')
    for snr in [0.01,0.05,0.07,0.1,0.2, 1,10]:
        folder_views = (f'{folder}/snr_{snr}')
        view, _, _, _ = generate_one_view(gt, np.array([0,0,0]), np.zeros(3), 3, ParametersDataGeneration(snr=snr))
        rotated_view, _ = rotation(view, get_rotation_matrix([90, 0, 0]))
        cut = rotated_view[25, :, :]
        save(f'/home/eloy/Documents/documents latex/these/images/test_noise/illustr_views/view_snr_{snr}.png', cut)

    """
    fold_gt = f'{PTH_GT}/2D'
    fold_cst = '/home/eloy/Documents/presentation_cst/illustr_mod_obs'
    folder_views = '/home/eloy/Documents/presentation_cst'

    generate_and_save_data(fold_cst, fold_gt, 'recepteurs_AMPA.tif',
                           ParametersDataGeneration(sig_xy=1, sig_z=20, nb_views=20, size=200, nb_dim=2, snr=7, partial_labelling=True,
                                                    nb_whole=70, std_min=0.001, std_max=0.005))

    1/0
    """



    1/0
    #pytorch3d.transforms.axis_angle_to_matrix(axis_angle: torch.Tensor)
    gt = read_image(f'{PTH_GT}/recepteurs_AMPA.tif')

    dilated_gt = dilate_one_dimension(gt, 8)
    print('shp', dilated_gt.shape)
    print(dilated_gt)
    save('./dilated_gt.tif', dilated_gt)
    1/0



    gt2 = read_image(f'{PTH_GT}/HIV-1-Vaccine_prep.tif')
    psf = crop_center(resize(gt + gt2, (20, 20, 20)), (50, 50, 50))
    psf/=np.sum(psf)
    convention = 'XYZ'
    params = ParametersDataGeneration(convention=convention, rot_vecs=[[45,180,0],[45,0,0]], nb_views=3, sig_xy=1, sig_z=5, snr=10000, order=3)
    rot = np.array([4,2,3])
    rot = rot/np.linalg.norm(rot)
    rot = [0,0,180]
    view1 = generate_one_view(gt2, rot, [0,0,0], 3, params)
    view2 = generate_one_view(gt2, [0,0,0], [0,0,0], 3, params)
    view2_rotated, _ = rotation(view2, get_rotation_matrix(rot, convention=convention))
    save(f'view1.tif', view1)
    save('view2.tif', view2)
    save('view_rotated.tif', view2_rotated)
    print(view1.real[25, 20:30, 20:30])
    print(view2_rotated.real[25, 20:30, 20:30])
    print('is close', np.isclose(view2_rotated.real, view1, atol=10**-2).sum()/(50**3))

    #generate_and_save_data('views', PTH_GT, 'recepteurs_AMPA.tif', params)
    1/0

    """
    from common_image_processing_methods.others import resize
    from manage_files.read_save_files import read_image


    fold_projections = '/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/projection_views'
    gt_names = ["recepteurs_AMPA", "HIV-1-Vaccine_prep", "clathrine", "emd_0680", "Vcentriole_prep"]
    for gt_name in gt_names:
        for nb_views in [100]:
            gt_path = f"/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{gt_name}.tif"
            gt_resized = resize(read_image(gt_path), (45,45,45))
            save('recepteurs_AMPA_resized.tif', gt_resized)
            1/0
            sub_fold = f'{fold_projections}/{gt_name}/nb_views_{nb_views}_no_anis'
            sub_fold_mrc = f'{sub_fold}_mrc'
            make_dir(sub_fold)
            make_dir(sub_fold_mrc)
            generate_and_save_data(sub_fold, nb_views, 0, 0, f"/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths", f'{gt_name}.tif',
                                   0,10, 3, 50, partial_labelling=False, projection=False)
            for fn in os.listdir(sub_fold):
                os.system(f'tif2mrc {sub_fold}/{fn} {sub_fold_mrc}/{fn}.mrc')

    1/0




    channel = 'c2'
    pth_views = f'/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/preprocessed_resize_ratio_2/{channel}'
    pth_projected_views = f'/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/' \
                          f'preprocessed_resize_ratio_2/{channel}_projected'
    make_dir(pth_projected_views)
    views ,names = read_images_in_folder(pth_views)
    for i in range(len(views)):
        projected = np.sum(views[i], axis=0)
        save(f'{pth_projected_views}/{names[i]}', projected)

    """


    from common_image_processing_methods.others import crop_center

    sigs_z = [5,10,15]
    sigs_xy = [1.5]
    gt_names = ["recepteurs_AMPA"]#, "HIV-1-Vaccine_prep", "clathrine", "emd_0680", "Vcentriole_prep"]
    crop_size = 90
    for gt_name in gt_names:
        gt_path = f"/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{gt_name}.tif"
        gt = io.imread(gt_path)
        gt = crop_center(gt, (crop_size,crop_size,crop_size))
        folder_psf = "/home/eloy/Documents/stage_reconstruction_spfluo/views/PSFs"
        make_dir(folder_psf)
        snr = 1000
        for sig_z in sigs_z:
            folder_views = f"/home/eloy/Documents/stage_reconstruction_spfluo/views/{gt_name}/single_view/sig_z_{sig_z}"
            make_dir(folder_views)
            for n in range(1):
                if gt_name == "emd_0680":
                    gt, _ = rotation(gt, get_rotation_matrix([90,0,0]))
                #gt = crop_center(gt, (80,80,80))
                gt[gt<0] = 0
                save(f'{folder_views}/gt.tif', gt)
                #gt = simulate_partial_labelling(gt, nb_whole=120, std_min=0.02, std_max=0.05)
                PSF, filtered = convolve_noise(gt, sig_z, 1, snr)
                filtered = simulate_partial_labelling(filtered, nb_whole=70)
                # save(f'{folder_psf}/sigz{sig_z}_sigxy{sig_xy}.tif', PSF)
                save(f'{folder_views}/view_{n}.tif', filtered)

    1/0

    
    folder_views = f"/home/eloy/Documents/stage_reconstruction_spfluo/"
    nb_tests_per_point = 1
    for ground_truth_name in ["emd_0680", "recepteurs_AMPA"]:
        for sig_z in [1,3,5,7,10,12,15,17,20]:
            generate_and_save_data(f'{folder_views}/{ground_truth_name}/sig_z_{sig_z}', 80, sig_z,
                                       1.5, "../../ground_truths", f"{ground_truth_name}.tif", sigma_trans_ker=0,
                                       snr=100, nb_dim=3, size=50, partial_labelling=False, projection=False)
    
    folder_views = f"/home/eloy/Documents/views"
    nb_tests_per_point = 10
    for ground_truth_name in ["recepteurs_AMPA", "HIV-1-Vaccine_prep", "clathrine", "emd_0680", "Vcentriole_prep"]:
        for nb_views in [60,70,80]:
            for sig_z in [5,10,15]:
                for t in range(nb_tests_per_point):
                    generate_and_save_data(f'{folder_views}/{ground_truth_name}/nb_views_{nb_views}/sig_z_{sig_z}/set_{t}', nb_views, sig_z, 1.5,"../../ground_truths", f"{ground_truth_name}.tif", sigma_trans_ker=0,
                               snr=100, nb_dim=3, size=50, partial_labelling=False, projection=False)








