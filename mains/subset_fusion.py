import numpy as np

from manage_files.paths import PATH_PROJECT_FOLDER
import os
from manage_files.read_save_files import read_csv, write_array_csv, read_image, save
from common_image_processing_methods.registration import translate_to_have_one_connected_component
from common_image_processing_methods.rotate_symmetry_axis import find_rot_mat_between_centriole_axis_and_z_axis
from common_image_processing_methods.rotation_translation import rotation, get_3d_rotation_matrix, get_rot_vec_from_3d_rot_mat

fold_results = f'{PATH_PROJECT_FOLDER}/results/splitted_in_subsets'
files = [f'set{i}' for i in range(1,11)]
rot_vecs = np.zeros((500, 3))
for sub_fold in ['set10']:
    print('sub fold', sub_fold)
    fd = f'{fold_results}/{sub_fold}'
    recons = read_image(f'{fd}/intermediar_results/recons_epoch_28.tif')
    translated = translate_to_have_one_connected_component(recons)
    save(f'{fd}/transalted_recons.tif', translated)
    #recons = read_image(f'{fd}/transalted_recons.tif')
    R = find_rot_mat_between_centriole_axis_and_z_axis(translated, axis_indice=0)
    rotated_im, _ = rotation(recons, R)
    save(f'{fd}/translated_aligned_recons.tif', rotated_im)
    sub_rot_vecs = read_csv(f'{fd}/intermediar_results/estimated_rot_vecs_epoch_30.csv')
    sub_rot_vecs_adjusted = np.zeros(sub_rot_vecs.shape)
    for g in range(len(sub_rot_vecs)):
        rm = get_3d_rotation_matrix(sub_rot_vecs[g])
        rv_adjusted = get_rot_vec_from_3d_rot_mat(rm@(R.T), 'zxz')
        sub_rot_vecs_adjusted[g, :] = rv_adjusted
    write_array_csv(sub_rot_vecs_adjusted, f'{fd}/rot_vecs_adjusted.csv')
    rot_vecs[50*i: 50*(i+1)] = sub_rot_vecs_adjusted

write_array_csv(rot_vecs, f'{fold_results}/all_rot_vecs.csv')





