import imageio
import numpy as np
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from skimage import io
import tifffile
from common_image_processing_methods.others import resize
#import mrcfile
import pickle


def save_in_file(txt, pth):
    f = open(pth, 'w')
    print(txt, file=f)


def load_pickle(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def read_4d(pth):
    return tifffile.imread(pth)

def save_pickle(tosave, save_fold, save_name):
    file = open(f'{save_fold}/{save_name}', "wb")
    pickle.dump(tosave, file)

def save_4d_for_chimera(im_4d, fn):
    """im_4d shape : (N, S, S, S). Use this to save a 4d image if you want to open it with chieraX"""
    all_NPC_image = np.array(im_4d).astype(np.float32)
    all_NPC_image_to_save = np.expand_dims(all_NPC_image,
                                           axis=2)  # (nbNPC, s, s, s) --> (nbNPC, s, 1, s, s) (so that chimerax can read)
    tifffile.imwrite(fn, all_NPC_image_to_save, imagej=True)
    print('shp', all_NPC_image_to_save.shape)

def read_multichannel(path):
    """read a multichannel image that has been saved from the convert_lif.py file"""
    im = read_image(path)
    im_c1 = im[[2 * x for x in range(len(im) // 2)]]
    im_c2 = im[[2 * x + 1 for x in range(len(im) // 2)]]
    im = np.array([im_c2, im_c1])
    return im

def read_image(path, mrc=False):
    if not mrc:
        return np.array(imageio.mimread(path, memtest=False))
    else:
        print('path', path)
        return mrcfile.read(path)


def return_het_with_name(fn):
    g = fn.split('_')
    if len(g) == 6:
        _, dil_val, rv1, rv2, rv3, _ = fn.split('_')
    else:
        dil_val = 10**10
    return np.float(dil_val)


def read_images_in_folder(fold, alphabetic_order=True, mrc=False, size=None, sort_fn=lambda x:x, multichannel=False):
    """read all the images inside folder fold"""
    files = os.listdir(fold)
    if alphabetic_order:
        files = sorted(files)
    files = sorted(files, key=sort_fn)
    images = []
    files_without_dir = []


    for fn in files:
        pth = f'{fold}/{fn}'
        if not os.path.isdir(pth):
            # print('pth', pth)
            if not multichannel:
                im = read_image(pth, mrc)
            else:
                im = read_multichannel(pth)
            if size is not None:
                im = resize(im, (size, size, size))
            images.append(im)
            files_without_dir.append(fn)
    return np.array(images), files_without_dir


def save_multi_channel(path, array):
    """save a multichannel image so that it can be read by imageJ"""
    transposed_array = np.transpose(array, (1,0,2,3)).astype(np.float32)
    tifffile.imwrite(path, transposed_array, imagej=True)

def save(path, array):
    # save with conversion to float32 so that imaej can open it
    io.imsave(path, np.float32(array))

def move_if_exists(src, dst):
    if os.path.exists(src):
        shutil.move(src, dst)


def make_dir(dir):
    """creates folder at location dir if it doesn't already exist"""
    if not os.path.exists(dir):
        print(f'directory {dir} created')
        os.makedirs(dir)


def delete_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def make_dir_and_write_array(np_array, fold, name):
    make_dir(fold)
    write_array_csv(np_array, f'{fold}/{name}')
    print(f'array saved at location {fold}, ith name {name}')


def write_array_csv(np_array, path):
    """write a 2d array into a csv file"""
    if len(np_array.shape) == 1:
        np.expand_dims(np_array, axis=0)
    pd.DataFrame(np_array).to_csv(path)


def read_csv(path, first_col=1, convert_float32=True):
    """read the csvfile at location 'path'. It reads only the colums after the column indexed by 'first_col'"""
    x = np.array(pd.read_csv(path))[:, first_col:].squeeze()
    if convert_float32:
        x = x.astype(np.float32)
    return x


def read_csvs(paths):
    contents = []
    for path in paths:
        contents.append(read_csv(path))
    return contents


def print_dictionnary_in_file(dic, file):
    """print dictionnary keys and attributes in a text file"""
    for at in list(dic.keys()):
        at_val = dic[at]
        print(f'{at} : {at_val}', file=file)


def save_figure(fold, save_name):
    make_dir(fold)
    plt.savefig(f'{fold}/{save_name}')
    print(f'figure saved at location {fold}, with name {save_name}')


def save_4d_vol_with_het(fold_in, pth_out):
    images, fns = read_images_in_folder(fold_in)
    images = np.array(images)
    hets = [float(fns[i].split('_')[-2]) for i in range(len(fns))]
    order = np.argsort(hets)
    images_sorted = images[order]
    images_sorted = np.expand_dims(images_sorted, axis=2)
    print('shp', np.shape(images_sorted))
    tifffile.imwrite(pth_out, images_sorted, imagej=True)
    return images_sorted


if __name__ == '__main__':
    from manage_files.paths import PATH_PROJECT_FOLDER
    pth = f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole/results_week_28_jannuary/test_het_0_unknown_rot_unknown_trans_GOOD'
    save_4d_vol_with_het(f'{pth}/ep_799/vols', f'{pth}/4d_recons.tiff')
    1/0

    pth = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views'

    save_4d_vol_with_het(f'{pth}/views_with_het_more_visible_symmetry_s_45/gt_dilated', f'{pth}/concat_het_300.tiff')
    1/0

    im = np.random.random((50,50,50))
    tifffile.imwrite('file.stk', im)
    im2 = tifffile.imread('file.stk')

    from manage_files.paths import *
    pth = f"{PATH_REAL_DATA}/Data_marine/selected_data/preprocessed_resize_ratio_2/c1"

    #read_images_in_folder(pth)

    from tifffile import imread
    pth = f'{PATH_REAL_DATA}/NPC/subtomograms_export.tif'
    im_stacked = read_image(pth)


    # a = np.random.random((6000, 30))
    # write_array_csv(a, './yy.csv')


