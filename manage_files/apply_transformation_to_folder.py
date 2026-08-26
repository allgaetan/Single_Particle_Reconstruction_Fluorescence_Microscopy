from manage_files.read_save_files import make_dir, save, write_array_csv
import os
from manage_files.paths import PATH_REAL_DATA
import numpy as np
from skimage import io
import imageio
from common_image_processing_methods.others import normalize, crop_center, transform_multichannel
from common_image_processing_methods.barycenter import center_barycenter
from manage_files.read_save_files import read_multichannel, save_multi_channel
from common_image_processing_methods.others import resize

def apply_transformations_to_image(im, funcs_to_apply, funcs_args, multi_channel=False):
    im = np.array(im)
    for idx, f in enumerate(funcs_to_apply):
        #print('apply function', f.__name__)
        if multi_channel:
            im = transform_multichannel(im, f, **(funcs_args[idx]))
        else:
            im = f(im, **(funcs_args[idx]))
    return im


def transform_folder(folder_in, folder_out, funcs_to_apply, funcs_args, folders_to_skip=[], multichannel=False):
    """
    apply all the funcion in the list functs_to_apply (in the same order)
    to all the images in folder_in and store the results in folder_out. Keep the same architecture of folder_in by recursively
    calling the function on each subfolders
    transformations f in funcs_to_apply take either  2D or 3D array as input, and some arguments in kwargs,
    and return either a 2D or 3D array
    transformation arguments are passed in func_args : func_args is a list of dictionnary, each of them containing the
    arguments of the correspoànding transformation
    transformations wont be applied to subfolders in folders_to_skip
    """
    make_dir(folder_out)
    files = [f for f in os.listdir(folder_in) if f not in folders_to_skip]
    print('f', files)
    for file in files:
        path = f'{folder_in}/{file}'
        if os.path.isdir(path):
            print(f'process folder {file}')
            transform_folder(path, f'{folder_out}/{file}', funcs_to_apply, funcs_args, folders_to_skip, multichannel=multichannel)
        else:
            if not multichannel:
                image = np.array(imageio.mimread(path))
            else:
                image = read_multichannel(path)
            im = apply_transformations_to_image(image, funcs_to_apply, funcs_args, multi_channel=multichannel)
            if not multichannel:
                save(f'{folder_out}/{file}', im)
            else:
                save_multi_channel(f'{folder_out}/{file}', im)


if __name__ == '__main__':
    #fold = f'{PATH_REAL_DATA}/Assembly_cropped_normalized'
    #fold_out = f'{PATH_REAL_DATA}/Assembly_cropped_normalized_centered'
    # fold = f'{PATH_REAL_DATA}/SAS6/picking/deconv_cropped_proto/good'
    # fold_out = f'{PATH_REAL_DATA}/SAS6/picking/deconv_cropped_proto/good_same_size_resized'
    fold = f'{PATH_REAL_DATA}/SAS6/picking/deconv_cropped_proto/top_views'
    fold_out = f'{PATH_REAL_DATA}/SAS6/picking/deconv_cropped_proto/top_views_centered'

    #transform_folder(fold, fold_out, [crop_center, normalize, resize], [{'size':(56,56,56)}, {}, {'desired_shape':(50,50,50)}], multichannel=True)
    transform_folder(fold, fold_out, [center_barycenter],
                     [{}], multichannel=True)