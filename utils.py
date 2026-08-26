import os
import matplotlib.pyplot as plt
import numpy as np
from manage_files.read_save_files import *
from manage_matplotlib.graph_setup import *
from skimage.morphology import dilation




def coordinates_main_diag(size):
    return [(i,i) for i in range(size)]


def coordinates_other_diagonal(size):
    return [(size-i, i-1) for i in range(1, size+1)]

def dilate(im, kernel_sz=None):
    if kernel_sz is None:
        sz = im.shape[0]//100
        kernel_sz = (sz, sz, sz)
    kernel = np.ones(kernel_sz, np.uint8)
    im = dilation(im, kernel)
    return im

def X_image(im_size):
    X = np.zeros((im_size, im_size))
    coordinates = coordinates_other_diagonal(im_size) + coordinates_main_diag(im_size)
    set_val_at_coordinates(X, coordinates, 255)
    return X


def set_val_at_coordinates(arr, coord, val):
    for p in coord:
        arr[p] = val

def Z_image(im_size):
    Z = np.zeros((im_size, im_size))
    coordinates = [(0, i) for i in range(im_size)] + coordinates_other_diagonal(im_size) + [(im_size-1, i) for i in range(im_size)]
    set_val_at_coordinates(Z, coordinates, 255)
    return Z


def Y_image(im_size):
    Y = np.zeros((im_size, im_size))
    coordinates = coordinates_other_diagonal(im_size)[im_size//2:] + coordinates_main_diag(im_size)[:im_size//2] + \
                  [(i, im_size//2) for i in range(im_size//2, im_size)]
    set_val_at_coordinates(Y, coordinates, 255)
    return Y


def concatenate_image(im1, im2, space_betwwen_ims):
    shp = (np.max([im1.shape[0], im2.shape[0]]), im1.shape[1] + im2.shape[1] + space_betwwen_ims)
    print('shp', shp)
    conc_im = np.zeros(shp)
    conc_im[0:im1.shape[0], 0:im1.shape[1]] = im1
    conc_im[0:im2.shape[0], im1.shape[1]+space_betwwen_ims:im1.shape[1] + im2.shape[1] + space_betwwen_ims] = im2
    return conc_im


def put_small_image_in_corner_of_big_image(small_image, big_image, start_soordinates):
    """
    if len(big_image.shape) == 3:
        big_image = big_image[0, :, :]
    if len(big_image.shape) == 4:
        big_image = big_image[0, :, :, 0]
    """
    big_image = big_image[0]
    print('shp', big_image.shape)
    h = small_image.shape[0]
    w = small_image.shape[1]
    big_image = big_image.astype(np.float32)
    if len(big_image.shape) == 3:
        for c in range(big_image.shape[2]):
            big_image[start_soordinates[0]:start_soordinates[0]+h, start_soordinates[1]:start_soordinates[1]+w, c] += small_image
    else:
        big_image[start_soordinates[0]:start_soordinates[0] + h, start_soordinates[1]:start_soordinates[1] + w] += small_image
    big_image[big_image >255] = 255
    return big_image


def find_indices_of_occurence(A, B):
    """A : Nd array. B : list of elements which are supposed to be included in A. Returns the indices of occurence
    of the elements of A in B"""
    idxs = []
    A = np.array(A)
    for b in B:
        idx = np.where(A==b)[0]
        idxs.append(idx[0])
    return idxs


def read_views_order_file(pth):
    f = open(pth)
    splited_str = f.readlines()[0].split(',')
    #splited_str[0] = splited_str[0][1:]
    #splited_str[-1] = splited_str[-1][:-1]
    for i in range(len(splited_str)):
        splited_str[i] = splited_str[i][2:-1]
    splited_str[-1] = splited_str[-1][:-2]
    return splited_str


def find_permutation_that_related_2_arrays(arr1, arr2):
    perm = []
    for i in range(len(arr1)):
        id = np.where(arr2 == arr1[i])
        perm.append(id)
    return np.array(perm)

def f(x):
    return 1,2,3


r = 8

if __name__ == '__main__':
    from manage_files.read_save_files import read_images_in_folder, make_dir, read_multichannel, read_4d
    from manage_files.paths import PATH_REAL_DATA, PATH_PROJECT_FOLDER
    from scipy.ndimage.interpolation import affine_transform
    from common_image_processing_methods.others import transform_multichannel, crop_with_given_center
    from common_image_processing_methods.registration import rotation_4d
    from common_image_processing_methods.rotation_translation import get_rotation_matrix, translation, get_3d_rotation_matrix, rotation
    from common_image_processing_methods.rotate_symmetry_axis import rotate_centriole_to_have_symmetry_axis_along_z_axis_4d, rotate_centriole_to_have_symmetry_axis_along_z_axis
    import shutil
    from reconstruction_with_dl.end_to_end_architecture_volume import plot_heterogeneity
    import numpy as np

    def f(x, b=True):
        if x>=0:
            return x
        else:
            if b:
                return 0
            else:
                return x


    vals = [f(x, True) for x in np.linspace(-100,100, 1000)]
    vals_ft = np.fft.fftn(vals)
    plt.plot(range(100), vals_ft[:100])
    plt.show()
    1/0


    pth = "/home/eloy/Documents/stage_reconstruction_spfluo/results_summary/voxels_reconstruction/coeff_kernel_rot/coeff_kernel_rot_30/test_5"
    recons_ampa = read_image(f'{pth}/recons_registered_gradient.tif')
    gt_ampa = read_image(f'{pth}/ground_truth.tif')
    rot_mat = get_rotation_matrix([90,0,0])
    rot_recons_ampa, _ = rotation(recons_ampa, rot_mat)
    rot_gt_ampa, _ = rotation(gt_ampa, rot_mat)

    fft_ampa = np.abs(np.fft.fftshift(np.fft.fftn(rot_recons_ampa[:,:,:])))
    fft_gt = np.abs(np.fft.fftshift(np.fft.fftn(rot_gt_ampa[:,:,:])))

    #fft_ampa[fft_ampa>5] = 0
    #fft_gt[fft_gt>5] = 0

    save(f'{pth}/recons_rotated.tif', rot_recons_ampa)
    save(f'{pth}/gt_rotated.tif', rot_gt_ampa)

    save(f'{pth}/recons_fft.tif', fft_ampa)
    save(f'{pth}/gt_fft.tif', fft_gt)
    1/0



    pth = "/home/eloy/Documents/documents latex/these/images/fluofire/impose_cylinder/results_real_data"
    im1 = read_4d(f'{pth}/4d_vol_channel_1.tiff')
    im2 = read_4d(f'{pth}/4d_vol_channel_2.tiff')
    im1_rot, R = rotate_centriole_to_have_symmetry_axis_along_z_axis_4d(im1, axis_indice=2)
    im2_rot = rotation_4d(im2, R)
    save_4d_for_chimera(im1_rot, f'{pth}/4d1_rot.tiff')
    save_4d_for_chimera(im2_rot, f'{pth}/4d2_rot.tiff')
    1/0

    pth_save = "/home/eloy/Documents/documents latex/these/images/fluofire/impose_cylinder"
    pth_results = f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole'
    pth_zero_pos = f'{pth_results}/impose_cylinder_zero_pos/ep_299'
    pth_non_zero_pos_diff_weight = f'{pth_results}/bicanal_donut_views_1_pos_param_r/ep_4999'
    pth_non_zero_pos_same_weight = f'{pth_results}/impose_cylinder_est_evth_r/ep_299'

    def plot_est_het(pth, pth_save, pos=0.3):
        size_text1 = 70
        size_text2 = 70
        line_width = 7
        s= 7
        make_dir(pth_save)
        est_het = read_csv(f'{pth}/est_heterogeneities.csv')
        true_het = read_csv(f'{pth}/true_heterogeneities.csv')
        y_labels = ["Longueur estimée canal 1", "Longueur estimée canal 2", "Rayon estimé canal 1", "Rayon estimé canal 2",
                    "Largeur estimée canal 1", "Largeur estimée canal 2", "Position canal 1"]

        set_up_graph(SMALLER_SIZE=size_text2, MEDIUM_SIZE=size_text1)
        plt.scatter(true_het, est_het[:, 0], c='blue', label="Longueur estimée canal 1", linewidth=s)
        plt.scatter(true_het, est_het[:, 1], c='green', label="Longeur estimée canal 2", linewidth=s)
        plt.plot(np.linspace(0.05, 1.5, 250), np.linspace(0.05, 1.5, 250), color="red", label="Longueur réelle canal 1", linewidth=line_width)
        x = 80
        true_lenght_second = list(np.linspace(0.05, 0.5, x)) + [0.5] * (250 - x)
        plt.plot(np.linspace(0.05, 1.5, 250), true_lenght_second, color="orange", label="Longueur réelle canal 2", linewidth=line_width)
        plt.xlabel("Longueur réelle canal 1")
        plt.grid()
        plt.legend()
        plt.savefig(f'{pth_save}/est_het_lenght.png')
        plt.close()

        set_up_graph(SMALLER_SIZE=size_text2, MEDIUM_SIZE=size_text1)
        plt.scatter(true_het, est_het[:, 2], c='blue', label="Rayon estimé canal 1", linewidth=s)
        plt.scatter(true_het, est_het[:, 3], c='green', label="Rayon estimé canal 2", linewidth=s)
        plt.plot(np.linspace(0.05, 1.5, 250), [0.5]*250, color="red", label="Rayon réel canal 1", linewidth=line_width)
        plt.plot(np.linspace(0.05, 1.5, 250), [0.15]*250, color="orange", label="Rayon réel canal 2", linewidth=line_width)
        plt.xlabel("Longueur réelle canal 1")
        plt.grid()
        plt.legend()
        plt.savefig(f'{pth_save}/est_het_radius.png')
        plt.close()

        set_up_graph(SMALLER_SIZE=size_text2, MEDIUM_SIZE=size_text1)
        plt.scatter(true_het, est_het[:, 4], c='blue', label="Largeur estimée canal 1", linewidth=s)
        plt.scatter(true_het, est_het[:, 5], c='green', label="Largeur estimée canal 2", linewidth=s)
        plt.plot(np.linspace(0.05, 1.5, 250), [0.1] * 250, color="red", label="Largeur réelle canal 1", linewidth=line_width)
        plt.plot(np.linspace(0.05, 1.5, 250), [0.04] * 250, color="orange", label="Largeur réelle canal 2", linewidth=line_width)
        plt.xlabel("Longueur réelle canal 1")
        plt.grid()
        plt.legend()
        plt.savefig(f'{pth_save}/est_het_width.png')
        plt.close()

        set_up_graph(SMALLER_SIZE=size_text2, MEDIUM_SIZE=size_text1)
        plt.scatter(true_het, est_het[:, 6], c='green', label="Position estimée canal 2", linewidth=s)
        plt.plot(np.linspace(0.05, 1.5, 250), [pos] * 250, color="orange", label="Position réelle canal 2", linewidth=line_width)
        plt.xlabel("Longueur réelle canal 1")
        plt.grid()
        plt.legend()
        plt.savefig(f'{pth_save}/est_het_pos.png')
        plt.close()
        """
        for l in range(est_het.shape[1]):
            est_het_l = est_het[:, l]
            # mean_est_het = [np.mean(est_het_l[true_het_l == i]) for i in range(1, m)]
            set_up_graph()
            plt.scatter(true_het, est_het_l)
            # plt.plot(range(1, m), mean_est_het, color='r', marker='X')
            plt.xlabel("Longueur réelle canal 1")
            plt.ylabel(y_labels[l])
            plt.grid()
            plt.savefig(f'{pth_save}/est_het_wrt_true_het_{l}.png')
            plt.close()
        """

    plot_est_het(pth_non_zero_pos_same_weight, f"{pth_save}/non_zero_pos_same_weight")
    plot_est_het(pth_non_zero_pos_diff_weight, f"{pth_save}/non_zero_pos_diff_weight")
    plot_est_het(pth_zero_pos, f"{pth_save}/zero_pos", pos=0)
    1/0

    pth = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views'
    pth_save = "/home/eloy/Documents/documents latex/these/images/fluofire/impose_symetry"
    for nb_missing_triplets in [1,2,3,4,5,8]:
        pth_gt = f'{pth}/{nb_missing_triplets}_missing_triplets_no_het/c1'
        im = read_image(f'{pth_gt}/gt/4d_vol_channel.tiff')
        rot = get_3d_rotation_matrix([0,90,0], convention='XYZ')
        im_rot, _ = rotation(im, rot)
        save_fold = f'{pth_save}/nb_missing_triplets_{nb_missing_triplets}'
        make_dir(save_fold)
        save(f'{save_fold}/gt.tif', im_rot)
    1/0



    pth = '/home/eloy/Documents/documents latex/these/images/fluofire/heterogene_recons_real_data'

    im = read_4d(f'{pth}/registered_centriole.tiff')
    im_cropped_4d = []
    for i in range(len(im)):
        im_cropped = crop_with_given_center(np.expand_dims(im[i],axis=0), [24,24,27], [50,50,50], [1,1,1])
        im_cropped_4d.append(im_cropped)
    im_cropped_4d = np.array(im_cropped_4d).squeeze()
    save_4d_for_chimera(im_cropped_4d, f'{pth}/registered_centered_recons.tiff')
    1/0

    im = read_4d(f'{pth}/recons.tiff')
    rotated_im = rotate_centriole_to_have_symmetry_axis_along_z_axis_4d(im, slice_idx=30)
    save_4d_for_chimera(rotated_im, f'{pth}/registered_centriole.tiff')
    1/0


    pth_est_het = f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole/diff_max_lenght/ep_1599/est_heterogeneities.csv'
    pth_true_het = f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole/diff_max_lenght/ep_1599/true_heterogeneities.csv'
    est_het = np.expand_dims(read_csv(pth_est_het), axis=1)
    true_het = read_csv(pth_true_het)
    print('sh', est_het.shape)
    print('rgr', true_het.shape)
    plot_heterogeneity(true_het, est_het, f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole/diff_max_lenght/ep_1599')
    1/0

    pth = (f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole/bicanal_donut_views_regul_pos_max_0.2_t5knwon_transrandom_order/'
           f'ep_9999/est_heterogeneities_sorted.csv')
    csv_file = read_csv(pth)
    second_het_val = csv_file[:, 1]
    fourth_het_val = csv_file[:, 3]
    sicth_het_val = csv_file[:, 5]
    idxs1 = np.where(second_het_val <= 0.1)
    print('t1', len(idxs1[0]))
    idxs2 = np.where(fourth_het_val<=0.13)
    print('t2', len(idxs2[0]))
    idxs3 = np.where(sicth_het_val<=0.0375)
    print('t3', len(idxs3[0]))

    1/0

    def f(x, a, b):
        return np.tanh(x-a) + np.tanh(-x+b)


    L = np.linspace(-10,10,2000)
    M = f(L, -4, 4)
    plt.plot(L, M)
    plt.show()
    1/0

    def fonction_sigmoide(x, a=0, b=1):
        """
        Fonction sigmoïde qui augmente progressivement de 0 à 1 sur tout l'intervalle réel.

        Arguments :
        x : float ou tableau numpy, point(s) où évaluer la fonction.
        a : float, paramètre de décalage horizontal (par défaut : 0).
        b : float, paramètre de contraction/étirement horizontal (par défaut : 1).

        Retourne :
        y : float ou tableau numpy, valeur(s) de la fonction sigmoïde évaluée en x.
        """
        return 1 / (1 + np.exp(-b * (x - a)))


    # Exemple d'utilisation :
    x_values = np.linspace(-5, 5, 100)
    y_values = fonction_sigmoide(x_values)

    import matplotlib.pyplot as plt

    plt.plot(x_values, y_values)
    plt.title("Fonction sigmoïde")
    plt.show()
    1/0

    pth = "/home/eloy/Documents/documents latex/these/images/real_data_heterogene_views/tubuline_sas6/tifs"
    for im_name in os.listdir(pth):

        pth_im = f'{pth}/{im_name}'
        im = read_multichannel(pth_im)
        print('shp', im.shape)
        for c in range(2):
            pth_c = f'{pth}/c{c+1}'
            make_dir(pth_c)
            im_c = im[c, :, :, :]
            save(f'{pth_c}/{im_name}', im_c)


    pth_root = f'{PATH_REAL_DATA}/SAS6/picking/deconv_cropped_proto'
    fold_views = f'{pth_root}/good_same_size_resized'
    fold_top_views = f'{pth_root}/top_views'
    fold_top_views_good = f'{fold_top_views}/good_top_views'
    fold_out = f'{pth_root}/duplicated_top_views'
    make_dir(fold_out)

    """
    filn = '190807 U2OS SAS6 488 Tub 568 bis (SAME THAN THE FROZEN FROM 3007).lif - Lightning 012Series012_Lng.tif_1'
    im = read_multichannel(f'{fold_top_views_good}/{filn}.tif')
    t_vec = [0, -5, -7]
    #t_vec = [0, 5,  -5]
    im_centered = transform_multichannel(im, translation, trans_vec=t_vec)
    save_multi_channel(f'{fold_top_views_good}/{filn}_centered.tif', im_centered)
    1/0
    """
    views, fns = read_images_in_folder(fold_views, multichannel=True, alphabetic_order=False)
    top_views, fns_top = read_images_in_folder(fold_top_views, multichannel=True, alphabetic_order=False)
    top_views_good, fns_top_good = read_images_in_folder(fold_top_views_good, multichannel=True, alphabetic_order=False)


    def rotation(volume, rot_mat, order=3, trans_vec=None):
        """apply a rotation around center of image"""
        if trans_vec is None:
            trans_vec = np.zeros(len(volume.shape))
        c = np.array([size // 2 for size in volume.shape])
        rotated = affine_transform(volume, rot_mat.T, c - rot_mat.T @ (c + trans_vec), order=order, mode='nearest')
        return rotated


    def duplicate_image_by_rotating(im, nb_duplic=4):
        ims_rotated = []
        for i in range(nb_duplic):
            rot_vec = [np.random.randint(360), 0, 0]
            rot_mat = get_rotation_matrix(rot_vec, 'XYZ')
            im_rotated = transform_multichannel(im, rotation, rot_mat=rot_mat)
            ims_rotated.append(im_rotated)
        return ims_rotated


    """
    nb_duplic = 4
    print('len top views', len(top_views))
    for v in range(len(top_views)):
        duplicated_views = duplicate_image_by_rotating(top_views[v], nb_duplic=nb_duplic)
        for i in range(nb_duplic):
            save_multi_channel(f'{fold_out}/{fns_top[v]}_duplic_{i}.tif', duplicated_views[i])
    """
    nb_duplic = 4
    for v in range(len(top_views_good)):
        duplicated_views = duplicate_image_by_rotating(top_views_good[v], nb_duplic=nb_duplic)
        for i in range(nb_duplic):
            save_multi_channel(f'{fold_out}/{fns_top_good[v]}_duplic_{i}.tif', top_views_good[v])
        save_multi_channel(f'{fold_out}/{fns_top_good[v]}', top_views_good[v])
    1 / 0

    feuille_calcul = read_csv(f'{PATH_PROJECT_FOLDER}/code/caracteristique images data set - Feuille 1.csv',
                              convert_float32=False, first_col=0)
    print('feuille', feuille_calcul.shape)
    print('first col', feuille_calcul[:, 0])
    indices_side = np.where(feuille_calcul[:, 0] == 'inclined-side')[0]

    print('indices side', len(indices_side))
    fns_side = feuille_calcul[:, 2][indices_side]
    fns_side = [f'{fns_side[i]}.tif' for i in range(len(fns_side))]
    print('fns_side len', len(fns_side))

    pth_root = f'{PATH_REAL_DATA}/SAS6/picking/deconv_cropped_proto'
    fold_views = f'{pth_root}/good_same_size_resized'
    fold_side_views = f'{pth_root}/inclined_views'
    make_dir(fold_side_views)
    for fn in os.listdir(fold_views):
        splitted_fn = fn.split('')
        # print('split', splitted_fn)
        new_fn = splitted_fn[0] + '\uf022' + splitted_fn[1]
        print('new fn', new_fn)
        if new_fn in fns_side:
            shutil.copyfile(f'{fold_views}/{fn}', f'{fold_side_views}/{fn}')
    1 / 0








    from manage_files.paths import PATH_RESULTS_SUMMARY

    pth = f'{PATH_RESULTS_SUMMARY}/voxels_reconstruction/coeff_kernel_rot'
    import shutil

    for c in [1,5,10,15,20,25,30,40,50,60,70,80,90,100]:
        for t in range(10):
            im_path = f'{pth}/coeff_kernel_rot_{c}/test_{t}'
            pth_views = f'{im_path}/intermediar_results'
            print('pth views', pth_views)
            if os.path.exists(pth_views):
                shutil.rmtree(pth_views)

    1 / 0






























    from manage_files.paths import PATH_PROJECT_FOLDER
    from common_image_processing_methods.others import normalize
    import tifffile

    import matplotlib.pyplot as plt
    import shutil

    """
    pth = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views'
    fd = f'{pth}/views_with_het_more_visible_symmetry_s_45_het_600/gt_dilated'
    images, fns = read_images_in_folder(fd)
    images = np.array(images)
    images = np.expand_dims(images, 2)
    print('shp', images.shape)
    images = (normalize(images) * (2 ** 8 - 1)).astype('uint8')
    # test = np.moveaxis(test,0,1)
    # test = np.random.rand((nc, nz, ny, nx))
    tifffile.imwrite(
        f'{PATH_PROJECT_FOLDER}/results_deep_learning/results_biologiste/recons_heterogeneity_alpha1_ep_1000.tiff',
        images, imagej=True)
    1/0

    pth = ('/home/eloy/Documents/stage_reconstruction_spfluo/results/'
           'gmm_test_nb_gaussians_2/clathrine/init_with_avg_of_views')
    sigs = [0.03,0.05,0.07,0.08,0.1,0.12]
    nb_gaussianss = [2,5,10,15,20,25,30,40,50,75,100,125,150,175,200,250]

    for sig in sigs:
        for nb_gaussians in nb_gaussianss:
            for t in range(10):
                fold = f'{pth}/sig_{sig}/nb_gaussians_{nb_gaussians}/test_{t}'
                shutil.rmtree(f'{fold}/intermediar_results')
    1/0






    np.random.seed(0)
    x = np.random.rand(50)
    y = np.random.rand(50)
    plt.scatter(x, y)
    plt.scatter(0.5, 0.5, color='red')
    plt.annotate('(0.5, 0.5)', (0.5, 0.5), textcoords="offset points", xytext=(0, -10), ha='center', color='red')
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.axhline(y=0.5, color='red', linestyle='--')
    plt.show()
    1/0

    symmetry_rot_mats = [get_rotation_matrix([0, 0, 360 * k / 9], 'zxz') for k in range(9)]

    im = read_image('view_113.21_.tif')
    rot_mat = symmetry_rot_mats[1]

    rotated, _ = rotation(im, rot_mat)
    save('rot.tif', rotated)

    1/0


    nc = 4
    nz = 50
    ny = 512
    nx = 512
    test = ((2 ** 8 - 1) * np.random.rand(nc, 1, nz, ny, nx)).astype('uint8')

    """
    images, fns = read_images_in_folder(f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogeneity_centriole/'
                                   f'test_known_rot_0/vols')
    """
    """
    images, fns = read_images_in_folder(f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogeneity_centriole/views/'
                                       f'variable_lenght/gt_dilated')
    """
     # metadata={'axes': 'ZCYX'})
    1/0

    import matplotlib.pyplot as plt
    pth = '/home/eloy/Documents/poster/results_illustrs'
    for n in ['gt_view', 'real_data_recons', 'recons_comparaison']:
        im = read_image(f'{pth}/{n}.png')
        print(im.shape)
        padded = np.pad(im, ((0,0), (50, 0), (0, 0), (0,0)))
        print('sh', padded.shape)
        save(f'{pth}/{n}_padded.png', padded[0])

    1/0
    """
    pth_im = "../ground_truths"
    im = read_image(f'{pth_im}/recepteurs_AMPA.tif').squeeze()
    print('im shape', im.shape)
    im = dilate(im, (5,5,5))
    plt.imshow(im.squeeze()[25])
    plt.show()
    1/0
    """


    space_btw_ims = 10

    for gt_name in ['clathrine', 'emd_0680', 'HIV_vaccine', 'recepteurs_AMPA']:
        fold = f"/home/eloy/Documents/stage_reconstruction_spfluo/article/illustrations/illustr_views/{gt_name}/ours/unknown_angles/partial_labelling"

        """
    cut_XY = read_image(f'{fold}/XY.png')
    sz = min(cut_XY.shape[1], cut_XY.shape[2]) // r
    XY_im = concatenate_image(X_image(sz), Y_image(sz), space_btw_ims)
    XY_im_with_XY = put_small_image_in_corner_of_big_image(dilate(XY_im), cut_XY, [5, 5])
    save(f'{fold}/XY_with_XY.png', XY_im_with_XY)
    """
    cut_XZ = read_image(f'{fold}/cut.png')
    sz = min(cut_XZ.shape[1], cut_XZ.shape[2]) // r
    XZ_im = concatenate_image(X_image(sz), Z_image(sz), space_btw_ims)
    print('cut XZ shape', cut_XZ.shape)
    XZ_im_with_XZ = put_small_image_in_corner_of_big_image(dilate(XZ_im, kernel_sz=(5,5)), cut_XZ, [5, 5])
    save(f'{fold}/cut_with_XZ.png', XZ_im_with_XZ)

    """
    cut_YZ = read_image(f'{fold}/YZ.png')
    sz = min(cut_YZ.shape[1], cut_YZ.shape[2]) // r
    YZ_im = concatenate_image(Z_image(sz), Y_image(sz), space_btw_ims)
    YZ_im_with_YZ = put_small_image_in_corner_of_big_image(dilate(YZ_im), cut_YZ, [5, 5])
    save(f'{fold}/YZ_with_YZ.png', YZ_im_with_YZ)
    """
    1/0





    for gt_name in ['clathrine', 'emd_0680', 'HIV_vaccine', 'recepteurs_AMPA']:
        pth_root = f'/home/eloy/Documents/stage_reconstruction_spfluo/article/illustrations/illustr_views/{gt_name}'
        pths = [f'{pth_root}/fortun']
        for pth in pths:
            pth_im = f'{pth}/cut.png'
            im = read_image(pth_im)
            sz = im.shape[1] // 12
            XZ_im = concatenate_image(X_image(sz), Z_image(sz), space_btw_ims)
            im_with_xz = put_small_image_in_corner_of_big_image(dilate(XZ_im), im, [5,5])
            save(f'{pth}/cut_with_xz.png', im_with_xz)
    1/0
    f = open('im_names', 'w')
    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/projection_views/clathrine/nb_views_100_mrc'
    fns = os.listdir(pth)
    for i,fn in enumerate(fns):
        os.rename(f'{pth}/{fn}', f'{pth}/view{i}.mrc')


    f = open('params', 'w')
    for i in range(3,100):
        f.write(f'00001@view{i}.mrc 3858.000000 3296.000000 5.378000 0.074578 200.000000 2.700000 0.100000 5000.0 5.0 0.0000 0.0000 54.970001 0.000000 1.000000 0.000000 \n')
    1/0



    """
    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/projection_views/recepteurs_AMPA/nb_views_20'
    pth_out = '/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/projection_views/recepteurs_AMPA/nb_views_20_mrc'
    make_dir(pth_out)
    files_names = os.listdir(pth)
    for fn in files_names:
        os.system(f'tif2mrc {pth}/{fn} {pth_out}/{fn}.mrc')
    1/0
    """


    pth_view = f'{pth_root}/views/c1'
    pths = [f'{pth_root}/RANSAC_recons/ab_init', f'{pth_root}/RANSAC_recons/ninefold_symmetry',
            f'{pth_root}/RANSAC_recons/refinement', f'{pth_root}/ours/nine_fold_symmetry',
            f'{pth_root}/ours/ab_init']

    for fold in pths:
        if fold == f'{pth_view}/other':
            r = 8
        else:
            r = 8
        cut_XY = read_image(f'{fold}/XY.png')
        sz = min(cut_XY.shape[1], cut_XY.shape[2]) // r
        XY_im = concatenate_image(X_image(sz), Y_image(sz), space_btw_ims)
        XY_im_with_XY = put_small_image_in_corner_of_big_image(dilate(XY_im), cut_XY, [5, 5])
        save(f'{fold}/XY_with_XY.png', XY_im_with_XY)

        cut_XZ = read_image(f'{fold}/XZ.png')

        sz = min(cut_XZ.shape[1], cut_XZ.shape[2]) // r
        XZ_im = concatenate_image(X_image(sz), Z_image(sz), space_btw_ims)
        XZ_im_with_XZ = put_small_image_in_corner_of_big_image(dilate(XZ_im), cut_XZ, [5, 5])
        save(f'{fold}/XZ_with_XZ.png', XZ_im_with_XZ)

        cut_YZ = read_image(f'{fold}/YZ.png')
        sz = min(cut_YZ.shape[1], cut_YZ.shape[2]) // r
        YZ_im = concatenate_image(Z_image(sz), Y_image(sz), space_btw_ims)
        YZ_im_with_YZ = put_small_image_in_corner_of_big_image(dilate(YZ_im), cut_YZ, [5, 5])
        save(f'{fold}/YZ_with_YZ.png', YZ_im_with_YZ)

    1/0

    for gt_name in ['clathrine', 'emd_0680', 'HIV_vaccine', 'recepteurs_AMPA']:
        pth_root = f'/home/eloy/Documents/article_reconstruction_micro_fluo/article/illustrations/illustr_views/{gt_name}'
        for sub_fold in ['gt', 'view']:
            fold = f'{pth_root}/{sub_fold}'

            cut_XY = read_image(f'{fold}/XY.png')
            sz = cut_XY.shape[1] // 12
            XY_im = concatenate_image(X_image(sz), Y_image(sz), space_btw_ims)
            XY_im_with_XY = put_small_image_in_corner_of_big_image(dilate(XY_im), cut_XY, [5,5])
            save(f'{fold}/XY_with_XY.png', XY_im_with_XY)


            cut_XZ = read_image(f'{fold}/XZ.png')
            sz = cut_XZ.shape[1] // 12
            XZ_im = concatenate_image(X_image(sz), Z_image(sz), space_btw_ims)
            XZ_im_with_XZ = put_small_image_in_corner_of_big_image(dilate(XZ_im), cut_XZ, [5, 5])
            save(f'{fold}/XZ_with_XZ.png', XZ_im_with_XZ)

            cut_YZ = read_image(f'{fold}/YZ.png')
            sz = cut_YZ.shape[1] // 12
            YZ_im = concatenate_image(Z_image(sz), Y_image(sz), space_btw_ims)
            YZ_im_with_YZ = put_small_image_in_corner_of_big_image(dilate(YZ_im), cut_YZ, [5, 5])
            save(f'{fold}/YZ_with_YZ.png', YZ_im_with_YZ)







    1/0

    """
    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/preprocessed_resize_ratio_2'
    pth_c1 = f'{pth}/c1_selection/best_selection'
    pth_c2 = f'{pth}/c2_selection/best_selection'
    ims_c1, fn1 = read_images_in_folder(pth_c1)
    ims_c2, fn2 = read_images_in_folder(pth_c2)
    print('fn1', fn1)
    print('fn2', fn2)
    pth_sum = f'{pth}/2_channels_sum'
    make_dir(pth_sum)

    for i in range(len(ims_c1)):
        im_c1 = ims_c1[i]
        im_c2 = ims_c2[i]
        sm = im_c1 + im_c2
        save(f'{pth_sum}/{fn1[i]}', sm)
    """
    XY = concatenate_image(X_image(30), Y_image(30), 5)
    im = read_image('cut.png')
    print(im.shape)
    modif_im = put_small_image_in_corner_of_big_image(XY, im, [5,5])
    print(modif_im.shape)
    save('modif_im.png', modif_im[0]"""







