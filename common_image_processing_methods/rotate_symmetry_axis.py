import numpy as np
from sklearn.decomposition import PCA
from manage_files.read_save_files import read_image, save
from common_image_processing_methods.rotation_translation import rotation
from common_image_processing_methods.others import normalize



def convert_im_to_point_cloud(im, thesh):
    coordinates = np.where(im>=thesh)
    coordinates = np.array(coordinates).T
    return coordinates

"""
def find_vector_orthogonal_to_two_vectors(Y, Z):
    y_1, y_2, y_3 = Y
    z_1, z_2, z_3 = Z
    return np.array([(y_3*z_2-z_3*y_2)/(y_2*z_1 - y_1 * z_2), (z_3*y_1 - y_3*z_1)/(y_2*z_1-y_1*z_2), 1])
"""


def skew_symmetric_cross_product(v):
    v1, v2, v3 = v[0], v[1], v[2]
    return np.array([[0, -v3, v2],
                     [v3, 0, -v1],
                     [-v2, v1, 0]])


def find_rotation_between_two_vectors(a, b):
    """returns the rotation matrix that rotates vector a onto vector b (the rotation matrix s.t. Ra = b)"""
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    ssc = skew_symmetric_cross_product(v)
    R = np.eye(3) + ssc + ssc.dot(ssc) * (1-c)/s**2
    return R


def find_centriole_symmetry_axis(centriole_im):
    ma = np.max(centriole_im) / 2
    centriol_pc = convert_im_to_point_cloud(centriole_im, ma / 3)
    pca = PCA(n_components=3)
    pca.fit(centriol_pc)
    #print('pca components', pca.components_)
    #print('sig values', pca.singular_values_)
    sum_of_2_by_2_differences = np.zeros(3)
    for i in range(3):
        for j in range(3):
            if j!=i:
                sum_of_2_by_2_differences[i] += np.abs(pca.singular_values_[i] - pca.singular_values_[j])
    idx_dim_pca = np.argmax(sum_of_2_by_2_differences)
    symmetry_axis = pca.components_[idx_dim_pca]
    return symmetry_axis


def find_rot_mat_between_centriole_axis_and_z_axis(centriole_im, axis_indice=0):
    symmetry_axis = find_centriole_symmetry_axis(centriole_im)
    z_axis = np.array([0, 0, 0])
    z_axis[axis_indice] = 1
    R = find_rotation_between_two_vectors(symmetry_axis, z_axis)
    return R


def rotate_centriole_to_have_symmetry_axis_along_z_axis_4d(centriole_im, axis_indice=0, slice_idx=0):
    R = find_rot_mat_between_centriole_axis_and_z_axis(centriole_im[slice_idx, :, :, :], axis_indice)
    rotated_im_4d = []
    for i in range(len(centriole_im)):
        rotated_im, _ = rotation(centriole_im[i, :, :, :], R)
        rotated_im_4d.append(rotated_im)
    return np.array(rotated_im_4d), R

def rotate_centriole_to_have_symmetry_axis_along_z_axis(centriole_im, axis_indice=0):
    R = find_rot_mat_between_centriole_axis_and_z_axis(centriole_im, axis_indice)
    rotated_im, _ = rotation(centriole_im, R)
    return rotated_im


if __name__ == '__main__':
    from manage_files.paths import PATH_PROJECT_FOLDER

    pth_fold = '/home/eloy/Documents/stage_reconstruction_spfluo/results_deep_learning/centriole/results_week_21_june/' \
               'views_with_het_more_visible_symmetry_s_45_2_unknown_rot_unknown_trans_impose_sym_GOOD/ep_109'
    pth_fold = '/home/eloy/Documents/stage_reconstruction_spfluo/results_deep_learning/centriole/results_week_28_june/' \
               'real_data_knwow_rot_2_unknown_trans/ep_179'
    pth_fold = f'{PATH_PROJECT_FOLDER}/results/real_data_centriole_fixed_poses/intermediar_results'
    #pth_fold = '/home/eloy/Documents/stage_reconstruction_spfluo/results_deep_learning/' \
     #     'results_week_14_june/variable_lenght_s_45_anis_3_alpha_1_rot_180_unknown_rot'


    im = read_image(f'{pth_fold}/recons_epoch_50.tif')
    im_rotated = rotate_centriole_to_have_symmetry_axis_along_z_axis(im, axis_indice=2)
    save(f'{pth_fold}/est_vol_rotated.tif', normalize(im_rotated))
    1/0


    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/results/cryo-RANSAC/recons_refined_registered.tif'
    pth_out = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/results/cryo-RANSAC/recons_refined_registered_2.tif'

    alphas = np.arange(0.1, 1, 0.1)
    for alpha in alphas:
        pth = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogeneity_centriole/isotrop_alpha/recons_ep_99.tif'
        pth_out = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogeneity_centriole/isotrop_alpha/recons_rotated.tif'
        im = read_image(pth)
        im_rotated = rotate_centriole_to_have_symmetry_axis_along_z_axis(im)
        save(pth_out, im_rotated)

