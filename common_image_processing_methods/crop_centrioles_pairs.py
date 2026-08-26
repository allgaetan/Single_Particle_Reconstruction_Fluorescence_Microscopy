import os

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC
from manage_files.read_save_files import read_image, save, save_multi_channel
from manage_files.paths import *
from common_image_processing_methods.barycenter import center_barycenter
from scipy.ndimage import gaussian_filter
from common_image_processing_methods.others import crop_center, resize_to_given_size_of_pixels, crop_with_given_center
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import gaussian_mixture_isotrop_identical_gaussians, make_grid
from common_image_processing_methods.otsu_thresholding import otsu_thresholding
from scipy.signal.windows import tukey as tukey_scipy
from scipy.ndimage import affine_transform
from common_image_processing_methods.rotate_symmetry_axis import find_rotation_between_two_vectors
import tifffile
from common_image_processing_methods.others import cylinder_mask


def keep_2_first_classes_dbscans(points, dbscan_labels, nb_kept_clusters=2, cluster_to_merge=None, cluster_to_remove=[]):
    """Points : list of points fitted by the algorithm dbscan
    dbscan : output of dbscan algorithm. List of labels associated to points. If labels[i] = -1, it means that
     the point was not associated to a cluster by dbscan.
     This function removes -1 labels, and keep the 2 clusteres having the most elements
     Optionaly, merge cluster indexed by given indices"""
    if cluster_to_merge is not None:
        for c in cluster_to_merge[1:]:
            dbscan_labels[np.where(dbscan_labels == c)[0]] = cluster_to_merge[0]
    for c in cluster_to_remove:

        dbscan_labels[np.where(dbscan_labels==c)[0]] = -1
    nb_clusters = np.max(dbscan_labels) + 1
    np_points_by_cluster = [len(np.where(dbscan_labels==c)[0]) for c in range(nb_clusters)]
    bigger_clusters = np.flip(np.argsort(np_points_by_cluster))[:nb_kept_clusters]
    indices_points_to_keep = []
    for c in bigger_clusters:
        indices_points_to_keep += list(np.where(dbscan_labels == c)[0])

    new_points = points[indices_points_to_keep]
    new_labels = dbscan_labels[indices_points_to_keep]

    new_new_labels = np.zeros(len(new_labels))
    for i,c in enumerate(bigger_clusters):
        new_new_labels[new_labels == c] = i
    # print(new_labels[5000:])
    return new_points, new_new_labels.astype(int), len(bigger_clusters)


def tukey(
    shape: tuple[int], alpha: float = 0.5, sym: bool = True
):
    tukeys = [tukey_scipy(s, alpha=alpha, sym=sym) for s in shape]
    tukeys_reshaped = [
        np.reshape(t, (1,) * i + (-1,) + (1,) * (len(shape) - i - 1))
        for i, t in enumerate(tukeys)
    ]
    final_window = tukeys_reshaped[0]
    for t in tukeys_reshaped[1:]:
        final_window = final_window * t
    return final_window


def one_crop_with_given_coordinates(im, coordinates, crop_size, pixel_size_z=3.39, ratio=2.2):
    im_cropped = im[:, int(coordinates[0]-crop_size//2*ratio):int(coordinates[0]+crop_size//2*ratio),
                 int(coordinates[1]-crop_size//2*ratio):int(coordinates[1]+crop_size//2*ratio)]
    pixel_size_before = [pixel_size_z, 1, 1]  # [3.39,1,1]
    pixel_size_after = np.array([1, 1, 1]) * ratio
    im_resized = resize_to_given_size_of_pixels(im_cropped, pixel_size_before, pixel_size_after)
    im_resize_cropped_z = im_resized[:44, :, :]
    im_resize_cropped_z = np.pad(im_resize_cropped_z, ((0,0),(16,16),(16,16)))
    return im_resize_cropped_z



t = 0
def save_clustered_binary_images(points, labels, shp, prefix_name='dbscan'):
    global t
    nb_clusters = np.max(labels) + 1
    dbscan_res = np.zeros(tuple([nb_clusters] + list(shp)))
    all_points = np.zeros(shp)
    for i in range(len(points)):
        lab = labels[i]
        # print('lab', lab)
        all_points[tuple(points[i].astype(int))] = 1
        if lab != -1:
            dbscan_res[lab][tuple(points[i].astype(int))] = 1
    save(f'{pth_deconv_cropped_proto}/all_points_{t}.tif', all_points)
    t+=1
    for c in range(nb_clusters):
        save(f'{pth_deconv_cropped_proto}/{prefix_name}_{c}.tif', dbscan_res[c])


def get_cluster_center(points, labels, value):
    return np.mean(points[np.where(labels == value)], axis=0).astype(int)


def separate_centrioles(
    im: np.ndarray,
    output_size: tuple[int, int, int],
    threshold_percentage: float = 0.5,
    scale: tuple[float, float, float] = (1, 1, 1),
    channel: int = 0,
    tukey_alpha: float = 0.1,
    cluster_to_remove = [],
        cluster_to_merge =[]
):
    return separate_centrioles_coords(
        im,
        (np.asarray(im.shape[-3:]) - 1) / 2,
        im.shape[-3:],
        output_size,
        scale = scale,
        threshold_percentage=threshold_percentage,
        channel=channel,
        tukey_alpha=tukey_alpha,
        cluster_to_remove=cluster_to_remove,
        cluster_to_merge = cluster_to_merge
    )


def separate_centrioles_coords(
    image: np.ndarray,
    pos: tuple[float, float, float],
    dim: tuple[float, float, float],
    output_size: tuple[float, float, float],
    *,
    scale: tuple[float, float, float] = (1, 1, 1),
    threshold_percentage: float = 0.5,
    channel: int = 0,
    tukey_alpha: float = 0.1,
    cluster_to_remove=[],
    cluster_to_merge = []
):

    if image.ndim > 3:
        multichannel = True
    else:
        multichannel = False
        image = image[None]

    # extract patch from image
    patch = crop_with_given_center(image, pos, dim, scale, subpixel=False)[channel]
    # Thresholding
    points_im = np.stack(np.nonzero(patch > np.max(patch) * threshold_percentage), axis=-1)

    patch_top_left_corner = np.asarray(pos) - np.asarray(dim) / 2
    points = points_im * scale # + patch_top_left_corner  # points in world space
    # Clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(points)
    save_clustered_binary_images(points_im, dbscan.labels_, patch.shape)
    new_points, new_dbscan_labels, nb_kept_clusters = keep_2_first_classes_dbscans(points, dbscan.labels_,
                                                                 nb_kept_clusters=2,
                                                                 cluster_to_merge=cluster_to_merge, cluster_to_remove=cluster_to_remove)
    save_clustered_binary_images(new_points, new_dbscan_labels, patch.shape, prefix_name='new_dbscan')
    center1 = get_cluster_center(new_points, new_dbscan_labels,0)
    patch1 = crop_with_given_center(
        image, center1, output_size, scale, subpixel=False
    )
    if nb_kept_clusters != 2:
        patch2 = np.zeros(patch1.shape)
        patches = [patch1, patch2]
    else:
        center2 = get_cluster_center(new_points, new_dbscan_labels, 1)
        dbscan_centers = np.array([center1, center2])
        # extract 2 patches around the centroids

        patch2 = crop_with_given_center(
            image, center2, output_size, scale, subpixel=False
        )

        save_multi_channel(f'{pth_data}/patch1.tiff', patch1)
        save_multi_channel(f'{pth_data}/patch2.tiff', patch2)

        # Separate clusters
        svc = SVC(kernel="linear")
        svc.fit(new_points, new_dbscan_labels)

        def proj(x):
            return np.dot(x, np.asarray(svc.coef_)[0]) + np.asarray(svc.intercept_)

        # Compute hyperplane for each patch
        s = np.max(patch1.shape[-3:])
        size_tukey = np.asarray([s, s, s]) * 3**0.5
        t = tukey(np.round(size_tukey).astype(int), alpha=tukey_alpha)

        # Compute hyperplane for each patch and apply tukey window
        patches = []
        for patch, pos_patch in zip([patch1, patch2], dbscan_centers):
            image_coords = np.stack(
                np.meshgrid(*[np.arange(s) for s in patch.shape[1:]], indexing="ij"),
                axis=-1,
            )
            patch_top_left_corner = np.asarray(pos_patch) - np.asarray(output_size) / 2
            image_coords = image_coords * scale + patch_top_left_corner  # to image space
            pos_patch_proj = proj(pos_patch)

            # Compute the tukey window
            v = np.asarray(svc.coef_[0])
            d = np.asarray(svc.intercept_) / np.linalg.norm(v)
            v = v / np.linalg.norm(v)
            unit = np.asarray([0, 1, 0])
            R = find_rotation_between_two_vectors(unit, v)

            H1 = np.eye(4)
            H1[:3, 3] = -np.asarray(scale) * np.round(size_tukey) / 2
            H1[[0, 1, 2], [0, 1, 2]] = np.asarray(scale)  # pixel to world space
            H2 = np.eye(4)  # world space
            H2[:3, :3] = R  # world space
            pos_tukey_plane = (
                R
                @ (
                    -np.sign(pos_patch_proj)
                    * np.asarray(scale)
                    * unit
                    * np.round(size_tukey)
                    / 2
                )
                / np.asarray(scale)
            )  # pixel space

            center_patch = np.asarray(pos_patch)  # world space
            patch_top_left_corner = (
                np.asarray(center_patch) - np.asarray(output_size) / 2
            )  # world space
            center_patch_proj = center_patch - v * (
                np.dot(center_patch, v) + d
            )  # world space
            center_patch_proj_patch_space = (
                center_patch_proj - patch_top_left_corner
            ) / np.asarray(
                scale
            )  # pixel space
            H3 = np.eye(4)
            H3[[0, 1, 2], [0, 1, 2]] = 1 / np.asarray(scale)
            H3[:3, 3] = center_patch_proj_patch_space - pos_tukey_plane
            H = H3 @ H2 @ H1
            t_rotated = affine_transform(t, np.linalg.inv(H), output_shape=patch.shape[-3:])
            patch = patch * t_rotated
            patches.append(patch)
    if not multichannel:
        patch1, patch2 = patch1[0], patch2[0]
    return np.array(patches)


def pad_to_given_shape(im, target_shape):
    nb_channels = im.shape[0]
    padded_im = []
    for c in range(nb_channels):
        pads = []
        for d in range(len(target_shape)):
            diff = target_shape[d] - im[c].shape[d]
            if diff > 0:
                pads.append((0, diff))
            else:
                pads.append((0,0))
        padded_channel = np.pad(im[c], pads)
        padded_im.append(padded_channel)
    return np.array(padded_im)



def crop_and_resize(im, center, dim, scale, final_size, cylinder=True):
    im_cropped = crop_with_given_center(im, center, dim, scale=(1,1,1))
    im_cropped_resized = []
    for c in range(im_cropped.shape[0]):
        resized = resize_to_given_size_of_pixels(im_cropped[c], pixel_size_before=scale, pixel_size_after=(1,1,1))
        if cylinder:
            resized = resized * cylinder_mask(resized.shape[1], resized.shape[0])
        resized, _ = crop_center(resized, final_size)
        im_cropped_resized.append(resized)
    im_cropped_resized = np.array(im_cropped_resized)
    print('im cropped resized shape', im_cropped_resized.shape)
    return im_cropped_resized, im_cropped


def separate_and_resize(im, scale, output_size, save_fold, im_name, threshold_percentage = 0.5, channel=1, tukey_alpha = 0.1,
                        cluster_to_remove=[], cluster_to_merge=[]):
    output_size_redim = (np.array(output_size)/np.array(scale)).astype(int)
    im1, im2 = separate_centrioles(im, output_size_redim, threshold_percentage=threshold_percentage, channel=channel,
                                   tukey_alpha=tukey_alpha, cluster_to_remove=cluster_to_remove, cluster_to_merge=cluster_to_merge)
    im1_resized = []
    im2_resized = []
    for c in range(im1.shape[0]):
        im1_resized_c = resize_to_given_size_of_pixels(im1[c], pixel_size_before=scale, pixel_size_after=(1,1,1))
        im2_resized_c = resize_to_given_size_of_pixels(im2[c], pixel_size_before=scale, pixel_size_after=(1,1,1))
        im1_resized.append(im1_resized_c)
        im2_resized.append(im2_resized_c)
    im1_resized = np.array(im1_resized)
    im2_resized = np.array(im2_resized)
    im1_resized_padded = pad_to_given_shape(im1_resized, output_size)
    im2_resized_padded = pad_to_given_shape(im2_resized, output_size)
    #im1_1, im2_2 = separate_centrioles(im2, output_size, threshold_percentage=threshold_percentage,
                                       #channel=channel, tukey_alpha=tukey_alpha)
    save_multi_channel(f'{save_fold}/{im_name}_1.tif', im1_resized_padded)
    save_multi_channel(f'{save_fold}/{im_name}_2.tif', im2_resized_padded)
    #save_multi_channel(f'{save_fold}/{im_name}_21.tif', im1_1)
    #save_multi_channel(f'{save_fold}/{im_name}_22.tif', im2_2)


if __name__ == '__main__':
    pth_data = f'{PATH_REAL_DATA}/SAS6/picking'
    pth_deconv = f'{pth_data}/deconv'
    pth_raw = f'{pth_data}/raw'
    pth_deconv_cropped_proto = f'{pth_data}/deconv_cropped_proto'

    #im_name = "191023_U2OS_SAS6-488_tubs-568bis FROZEN FROM 300719.lif - Lightning 004Series004_Lng.tif"
    #im_name = "190805 U2OS SAS6 488 Tub 568 (SAME THAN THE FROZEN FROM 3007).lif - Lightning 008Series008_Lng.tif"
    #im_name = "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 020Series020_Lng.tif"
    #im_name = "190805 U2OS SAS6 488 Tub 568 (SAME THAN THE FROZEN FROM 3007).lif - Lightning 005Series005_Lng.tif"
    # im_name = "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 004Series004_Lng_2.tif"
    #im_name = "190315_U2OS 2X AA FA Sas6 488 Tub rabbit.lif - Lightning 007Series006_Lng.tif"
    # im_name = "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 014Series014_Lng_2.tif"
    # im_name = "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 002Series002_Lng_2.tif"
    # im_name = "190805 U2OS SAS6 488 Tub 568 (SAME THAN THE FROZEN FROM 3007).lif - Lightning 018Series019_Lng_2.tif"
    #im_name = "190807 U2OS SAS6 488 Tub 568 bis (SAME THAN THE FROZEN FROM 3007).lif - Lightning 013Series013_Lng.tif"
    im_name = "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 024Series025_Lng.tif"

    im = read_image(f'{pth_deconv}/{im_name}')
    im_c1 = im[[2 * x for x in range(len(im) // 2)]]
    im_c2 = im[[2 * x + 1 for x in range(len(im) // 2)]]
    im = np.array([im_c1, im_c2])

    first_slice = 14
    end_slice = 29
    pos = [(first_slice + end_slice)//2, 41, 25]
    #sz1 = 24
    sz2 = 55
    dims = [end_slice - first_slice, 29,29]
    scale = np.array([3.35, 1, 1])
    im_cropped_resized, im_cropped = crop_and_resize(im, pos, dims, scale, (sz2,sz2,sz2), cylinder=False)
    sv_pth = f'{pth_deconv_cropped_proto}/cropped'
    save_multi_channel(f'{sv_pth}/{im_name}', im_cropped_resized)
    save_multi_channel(f'{sv_pth}/non_resized/{im_name}', im_cropped)
    1/0




    fns = os.listdir(f'{pth_deconv_cropped_proto}/good')
    trop_dur = ["190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 024Series025_Lng.tif",
                "190807 U2OS SAS6 488 Tub 568 bis (SAME THAN THE FROZEN FROM 3007).lif - Lightning 013Series013_Lng.tif",
                "190315_U2OS 2X AA FA Sas6 488 Tub rabbit.lif - Lightning 013Series003_Lng.tif",
                "191023_U2OS_SAS6-488_tubs-568bis FROZEN FROM 300719.lif - Lightning 009Series010-1.tif",
                "190805 U2OS SAS6 488 Tub 568 (SAME THAN THE FROZEN FROM 3007).lif - Lightning 014Series016_Lng-1_2.tif",
                "191023_U2OS_SAS6-488_tubs-568bis FROZEN FROM 300719.lif - Lightning 004Series004_Lng.tif",
                "190805 U2OS SAS6 488 Tub 568 (SAME THAN THE FROZEN FROM 3007).lif - Lightning 008Series008_Lng.tif",
                "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 020Series020_Lng.tif",
                "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 005Series005_Lng.tif",
                "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 015Series015_Lng.tif",
                "1903022_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 013Series013_Lng_2.tif",
                "190805 U2OS SAS6 488 Tub 568 (SAME THAN THE FROZEN FROM 3007).lif - Lightning 005Series005_Lng.tif",
                "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 004Series004_Lng_2.tif",
                "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 016Series016_Lng.tif",
                "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 007Series007_Lng.tif",
                "190315_U2OS 2X AA FA Sas6 488 Tub rabbit.lif - Lightning 007Series006_Lng.tif",
                "190315_U2OS 2X AA FA Sas6 488 Tub rabbit.lif - Lightning 008Series009_Lng_2.tif",
                "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 014Series014_Lng_2.tif",
                "1903015_U2OS 2X AA FA Sas6 488 Tub rabbit.lif - Lightning 007Series006_Lng-1.tif",
                "191023_U2OS_SAS6-488_tubs-568bis FROZEN FROM 300719.lif - Lightning 003Series003-1.tif",
                "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 002Series002_Lng_2.tif",
                "191023_U2OS_SAS6-488_tubs-568bis FROZEN FROM 300719.lif - Lightning 006Series006_Lng.tif",
                "1903015_U2OS 2X AA FA Sas6 488 Tub rabbit.lif - Lightning 013Series003_Lng.tif",
                "190322_U2OS 2X AA FA Sas6 488 Tub rabbit bis.lif - Lightning 013Series013_Lng.tif",
                "190805 U2OS SAS6 488 Tub 568 (SAME THAN THE FROZEN FROM 3007).lif - Lightning 018Series019_Lng_2.tif"]

    do_not_process = [fn[:-6] for fn in fns] + trop_dur
    # print('dd', do_not_process)
    # for im_name in ["190315_U2OS 2X AA FA Sas6 488 Tub rabbit.lif - Lightning 001Series001_Lng.tif"]:
    for im_name in os.listdir(pth_deconv):
        if im_name not in do_not_process:
            print('im name', im_name)
            im = read_image(f'{pth_deconv}/{im_name}')
            im_c1 = im[[2*x for x in range(len(im)//2)]]
            im_c2 = im[[2*x+1 for x in range(len(im)//2)]]
            im = np.array([im_c1, im_c2])
            scale = np.array([3.35,1,1])
            save_multi_channel(f'{pth_deconv_cropped_proto}/original_image.tif', im)
            eps = 3
            min_samples = 100
            separate_and_resize(im, scale, (55, 55, 55), pth_deconv_cropped_proto, im_name, threshold_percentage=0.2,
                                                                                cluster_to_remove=[], cluster_to_merge=[])
            #print(f"did not work for {im_name}")
            print('nb processed', len(do_not_process))
            1/0

        else:
            print(f"i did not process {im_name}")
    #1/0


    """
    #centers = np.array([[13,45,41], [10,27,76]])
    centers = np.array([[22,43,31], [9,66,41]])
    im_c2_otsu = otsu_thresholding(im_c2, 0.1)
    save(f'{pth_data}/im_otsu.tif', im_c2_otsu)


    g = separate_centrioles(im, (50,50,50), threshold_percentage=0.5, channel=1, scale=(1,1,1))
    save_multi_channel(f'{pth_data}/im2_jean.tif', g[1].astype(np.float32))
    save_multi_channel(f'{pth_data}/im1_jean.tif', g[0].astype(np.float32))
    """





    pth_data = '/data/eloy/Assembly_tif/deconv/c1'
    # im_name = "190207_U2OS_CPAP_Tub345 ter_1"
    # im_name = "190131U2OS_CPAPendo_Tub345_3"
    #im_name = "190207_U2OS_CPAP_Tub345_8"
    # im_name = "190207_U2OS_CPAP_Tub345 ter_2"
    # im_name = "190305_U2OS 2X AA FA Cep135 488 Tub 568 four_1"
    #im_name = "190305_U2OS 2X AA FA Cep135 488 Tub 568 four_3"
    # im_name = "190207_U2OS_CPAP_Tub345_21"
    # im_name = "190207_U2OS_CPAP_Tub345_22"
    #im_name = "190207_U2OS_CPAP_Tub345_11"
    #im_name = "190207_U2OS_CPAP_Tub345_17"
    #im_name = "190207_U2OS_CPAP_Tub345_18"
    #im_name = "190207_U2OS_CPAP_Tub345_19"
    #im_name = "190207_U2OS_CPAP_Tub345_20"
    #im_name = "190207_U2OS_CPAP_Tub345_16"
    # im_name = "190207_U2OS_CPAP_Tub345 ter_3"
    # im_name = "190208_U2OS_POC1b_Tub345_2"
    #im_name = "190208_U2OS_POC1b_Tub345 bis_1"
    #im_name = "190208_U2OS_POC1b_Tub345 bis_3"
    #im_name = "190208_U2OS_POC1b_Tub345 bis_4"
    #im_name = "190208_U2OS_POC1b_Tub345 bis_7"
    #im_name = "190208_U2OS_POC1b_Tub345 bis_8"
    #im_name = "190208_U2OS_POC1b_Tub345 ter_1"
    # im_name = "190207_U2OS_CPAP_Tub345 bis_1_cropp2_cropp1"
    #im_name = "190208_U2OS_POC1b_Tub345 ter_6"
    #im_name = "190208_U2OS_POC1b_Tub345 ter_7"
    #im_name = "190208_U2OS_POC1b_Tub345 ter_10"
    #im_name = "190208_U2OS_POC1b_Tub345 ter_12"
    #im_name = "190208_U2OS_POC1b_Tub345 ter_16"
    #im_name = "190214_U2OS_POC5_Tub345_1"
    #im_name = "190214_U2OS_POC5_Tub345 bis_1"
    #im_name = "190214_U2OS_POC5_Tub345 bis_3"
    #im_name = "190214_U2OS_POC5_Tub345 bis_4"
    #im_name = "190214_U2OS_POC5_Tub345 bis_5"
    #im_name = "190214_U2OS_POC5_Tub345 bis_8"
    # im_name = "190215_U2OS_POC19(1)_Tub345_1_cropp1"
    #im_name = "190215_U2OS_POC19(1)_Tub345_4"
    im_name = "190301_U2OS 2X AA FA Cep135 488 Tub 568_12"
    im_name = "190301_U2OS 2X AA FA Cep135 488 Tub 568 bis_3"
    im_name = "190301_U2OS 2X AA FA Cep135 488 Tub 568 bis_7"
    im_name = "190305_U2OS 2X AA FA Centrin 488 Tub 568_2"
    im_name = "190305_U2OS 2X AA FA Centrin 488 Tub 568_3"
    im_name = "190305_U2OS 2X AA FA Centrin 488 Tub 568_5"
    im_name = "190305_U2OS 2X AA FA Centrin 488 Tub 568_8"
    im_name = "190315_U2OS 2X AA FA Sas6 488 Tub rabbit_3"
    pth_im = f'{pth_data}/{im_name}.tif'
    im = read_image(pth_im)

    #crop_size = (52,200,200)
    crop_hand = True
    resize = True
    if resize:
        ratio = 2.2
        pixel_size_z = 3.39
        crop_size = (13, 13, 13)
    else:
        ratio = 1
        pixel_size_z = 1
        crop_size = (44,44,44)
    if not crop_hand:
        im1, im2 = separate_centrioles(im[:, :, :], crop_size=crop_size, threshold_percentage=0.3, sigma_filter=20, pixel_size_z=pixel_size_z, ratio=ratio)
    else:
        im1 = one_crop_with_given_coordinates(im, (112,215), crop_size[0], pixel_size_z=pixel_size_z, ratio=ratio)
    fold_out = f'{PATH_PROJECT_FOLDER}/real_data/Assembly_cropped'
    save(f'{fold_out}/{im_name}_cropp1.tif', im1)
    save(f'{fold_out}/{im_name}_cropp2.tif', im2)
