import os

import numpy as np
import scipy.ndimage as scp
from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import make_grid

def threshold(im, thresh_val):
    im_out = (im>=thresh_val) * im
    return im_out


def transform_multichannel(im, transform, **transform_args):
    nb_channels = im.shape[0]
    im_transformed = []
    for c in range(nb_channels):
        im_c_transformed = transform(im[c], **transform_args)
        im_transformed.append(im_c_transformed)
    return np.array(im_transformed)


def crop_center(image, size, cylinder=False):
    """crop a sub array of shape 'size' at the center of the image. It pads the images at the dimensions where the shape is smaller
    than the crop size"""
    nb_dim = len(image.shape)
    padding = []
    for d in range(nb_dim):
        if image.shape[d] < size[d]:
            padding_d = int((size[d] - image.shape[d])//2+1)
            padding.append((padding_d, padding_d))
        else:
            padding.append((0,0))
    padding = tuple(padding)
    image = np.pad(image, padding, mode="constant", constant_values=0)
    size = np.array(size)
    c = [image.shape[0]//2, image.shape[1]//2, image.shape[2]//2]
    cropped = image[c[0]-size[0]//2:c[0]+size[0]//2, c[1]-size[1]//2:c[1]+size[1]//2, c[2]-size[2]//2:c[2]+size[2]//2]
    return cropped


def cylinder_mask(size_xy, size_z):
    grid = make_grid(size_xy, 2)
    distances_to_center = np.sqrt(grid[:,:,0] ** 2 + grid[:,:,1] ** 2)
    mask = (distances_to_center <= 1)
    mask_expanded = np.expand_dims(mask, axis=0)
    mask_expanded = np.repeat(mask_expanded, size_z, axis=0)
    return mask



def crop_with_given_center(
    image_data: np.ndarray,
    pos: tuple[float, float, float],
    dim: tuple[float, float, float],
    scale: tuple[float, float, float],
    subpixel: bool = False,
):
    """shape of image data : (N_c, N_x, N_y, N_z)"""
    def world_to_data_coord(pos):
        return pos / np.asarray(scale)

    pos = world_to_data_coord(np.asarray(pos))  # World coordinates to data coords
    box_size_world = np.asarray(dim, dtype=float)
    box_size_data = np.round(world_to_data_coord(box_size_world)).astype(np.int64)
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, 3] -= box_size_data / 2
    C = image_data.shape[0]
    particle_data = np.empty((C,) + tuple(box_size_data), dtype=image_data.dtype)
    if not subpixel:
        top_left_corner = np.round(mat[:3, 3]).astype(int)
        bottom_right_corner = top_left_corner + box_size_data
        xmin, ymin, zmin = top_left_corner
        xmax, ymax, zmax = bottom_right_corner
        original_shape = image_data.shape[1:]
        print('shp', original_shape)
        x_slice = slice(max(xmin, 0), min(xmax, original_shape[0]))
        y_slice = slice(max(ymin, 0), min(ymax, original_shape[1]))
        z_slice = slice(max(zmin, 0), min(zmax, original_shape[2]))
        x_overlap = slice(
            max(0, -xmin), min(box_size_data[0], original_shape[0] - xmin)
        )
        y_overlap = slice(
            max(0, -ymin), min(box_size_data[1], original_shape[1] - ymin)
        )
        z_overlap = slice(
            max(0, -zmin), min(box_size_data[2], original_shape[2] - zmin)
        )

    for c in range(C):
        padded_array = np.zeros(tuple(box_size_data), dtype=image_data.dtype)
        padded_array[x_overlap, y_overlap, z_overlap] = image_data[
            c, x_slice, y_slice, z_slice
        ]
        particle_data[c] = np.asarray(padded_array)

    return particle_data


"""
def crop_center_with_given_center(image, size, given_center):
    print('size', size)
    nb_dim = len(given_center)
    slices_crop = []
    for d in range(nb_dim):
        distance_left = given_center[d]
        distance_right = image.shape[d] - given_center[d]
        half_sz_crop = min(distance_left, distance_right)
        # s_crop = slice(max(0, given_center[d] - size[d]//2), min(given_center[d] + size[d] - size[d]//2, image.shape[d]))
        s_crop = slice(given_center[d]-half_sz_crop, given_center[d]+half_sz_crop)
        slices_crop.append(s_crop)
    cropped_image = image[slices_crop[0], slices_crop[1], slices_crop[2]]
    padded = np.zeros(tuple(size))
    slices_pad = []
    for d in range(nb_dim):
        sh = cropped_image.shape[d]
        s_pad = slice(size[d]//2 - sh//2, size[d]//2 + sh - sh//2)
        slices_pad.append(s_pad)
    print('padded shape', padded.shape)
    print('cropped image shape', cropped_image.shape)
    padded[slices_pad[0], slices_pad[1], slices_pad[2]] = cropped_image
    return padded, cropped_image
"""

def normalize(arr, min=0, max=1):
    norm_0_1 = (arr - np.min(arr))/(np.max(arr)-np.min(arr))
    norm = (max-min) * norm_0_1 + min
    return norm


def resize(in_array, desired_shape):
    """zoom an image to a desired shape"""
    array_shape = np.array(list(in_array.shape))
    desired_shape = np.array(desired_shape)
    return scp.zoom(in_array, desired_shape / array_shape, order=5)


def resize_to_given_size_of_pixels(volume, pixel_size_before, pixel_size_after):
    pixel_size_before, pixel_size_after = np.array(pixel_size_before), np.array(pixel_size_after)
    shape_before = volume.shape
    ratio_shape = pixel_size_before/pixel_size_after
    shape_after = shape_before * ratio_shape
    shape_after = [int(shape_after[d]+0.5) for d in range(len(pixel_size_after))]
    return resize(volume, shape_after)


def window_fft(im, size, sig_noise):
    fft_im = np.fft.fftshift(np.fft.fftn(im))
    window_fft_im = np.random.normal(0, sig_noise, im.shape)
    c = [im.shape[0] // 2, im.shape[1] // 2, im.shape[2] // 2]
    window_fft_im[c[0]-size[0]//2:c[0]+size[0]//2, c[1]-size[1]//2:c[1]+size[1]//2, c[2]-size[2]//2:c[2]+size[2]//2] \
        = crop_center(fft_im, size) - window_fft_im[c[0]-size[0]//2:c[0]+size[0]//2, c[1]-size[1]//2:c[1]+size[1]//2, c[2]-size[2]//2:c[2]+size[2]//2]
    return np.fft.ifftn(np.fft.ifftshift(window_fft_im))


def projection_z_axis(im):
    """project image along z axis"""
    return np.sum(im, axis=0)


def pad_all_channels_of_image(im, pad_vals):
    nb_channels = im.shape[2]
    shape_padded = np.shape(np.pad(im[:,:,0], pad_vals))
    im_padded = np.zeros((*shape_padded, 3))
    for c in range(nb_channels):
        im_c_padded = np.pad(im[:,:,c], pad_vals)
        im_padded[:,:,c] = im_c_padded
    print('shape', im_padded.shape)
    return im_padded





if __name__ == '__main__':
    import imageio
    from manage_files.read_save_files import save, make_dir, read_image, read_images_in_folder
    from manage_files.paths import *
    nb_views = 20
    pth_pt = '/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/projected_views_2/clathrine'
    pth = f'{pth_pt}/nb_views_{nb_views}'
    x = 300

    big_im = np.zeros((np.int(np.ceil(nb_views/5))*x+50, 5*x+50))
    ims, fns = read_images_in_folder(pth)
    for i,im in enumerate(ims):
        r = i//5
        c = i%5
        g = int(x*(r+0.5))
        h = int(x*(c+0.5))
        big_im[g:g+50, h:h+50] = im
    pth_big_im = f'{pth_pt}/big_im_{nb_views}_views'
    make_dir(pth_big_im)
    save(f'{pth_big_im}/big_im.tif', big_im)
    os.system(f'tif2mrc {pth_big_im}/big_im.tif {pth_big_im}/big_im.mrc')
    1/0

    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/cryo-SPARC_projects/empiar_10025_subset'
    cropped_path = '/home/eloy/Documents/stage_reconstruction_spfluo/cryo-SPARC_projects/empiar_10025_subset_cropped'
    make_dir(cropped_path)
    print('openned')
    files = os.listdir(pth)
    for fn in files:
        pth_im = f'{pth}/{fn}'
        im = read_image(pth_im)
        im_cropped = crop_center(im, (38,1000,1000))
        save(f'{cropped_path}/{fn}', im_cropped)

    im = read_image('/home/eloy/Documents/stage_reconstruction_spfluo/cryo-SPARC_projects/norm-amibox05-0.mrc')
    save(f'{cropped_path}/norm-amibox05-0.mrc', crop_center(im, (1000,1000)))

    1/0

    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/raw_splitted_channel/c1/211214_siCT_CEP164.lif - Series002.tif'
    central_slice = read_image(pth)
    print('sd', central_slice.shape)
    padded = crop_center(central_slice, (50,50,50))
    save('padded_linear_ramp.tif', padded)


    fold = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Data_marine/selected_data/preprocessed_resize_ratio_2'
    c1_pth = f'{fold}/c1'
    c2_pth = f'{fold}/c2'
    images_c1, files = read_images_in_folder(c1_pth)
    images_c2, _ = read_images_in_folder(c2_pth)
    folder_2_channels = f'{fold}/2_channels'
    make_dir(folder_2_channels)
    for i in range(len(images_c1)):
        im_c1 = images_c1[i]
        im_c2 = images_c2[i]
        size = im_c1.shape[0]
        im_2_channels = np.zeros((size, size, size, 2))
        im_2_channels[:,:,:,0] = im_c1
        im_2_channels[:,:,:,1]  = im_c2
        save(f'{folder_2_channels}/{files[i]}', im_2_channels)

    1/0



    pths = ["~/Documents/article_reconstruction_micro_fluo/article/illustrations/illustr_views/emd_0680/ransac/80_views"]
    for pth in pths:
        im = read_image(f'{pth}/chimera.png')
        print(im.shape)
        im = im[0, :, :, :]
        print(im.shape)
        pad_size = im.shape[0]//10
        im_padded = pad_all_channels_of_image(im, ((0,0), (17,17)))
        save(f'{pth}/chimera_padded.png', im_padded)

    1/0


    pth = f'{PATH_REAL_DATA}/Data_marine_raw'
    out_pth = f'{PATH_REAL_DATA}/Data_marine_raw_splitted_channel_2'
    make_dir(out_pth)
    make_dir(f'{out_pth}/c1')
    make_dir(f'{out_pth}/c2')
    for fn in os.listdir(pth):
         loc = f'{pth}/{fn}'
         if not os.path.isdir(loc):
            print(fn)
            im = read_image(loc)
            even_indices = [2*k for k in range(im.shape[0]//2)]
            odd_indices = [2*k+1 for k in range(im.shape[0]//2)]
            im_c1 = im[even_indices, :, :]
            im_c2 = im[odd_indices, :, :]
            save(f'{out_pth}/c1/{fn}', im_c1)
            save(f'{out_pth}/c2/{fn}', im_c2)


    1/0


    im = read_image(f'{pth}.tif')
    im[im<0.5] = 0
    im = normalize(im)
    save(f'{pth}_2.tif', im)
    1/0


    pth  = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Assembly_tif/raw/centrioles_to_pick'
    out_path = "/home/eloy/Documents/stage_reconstruction_spfluo/real_data/Assembly_tif/raw/centrioles_to_pick_selection_2"
    make_dir(out_path)
    for im_name in os.listdir(pth):
        print('im name', im_name)
        im = imageio.mimread(f'{pth}/{im_name}')
        im = np.array(im)
        if im.shape[0] < 30:
            save(f'{out_path}/{im_name}', im)



    """
    pth = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/high_res_images/raw/psf.tif'
    pth_save = '/home/eloy/Documents/stage_reconstruction_spfluo/real_data/high_res_images/psf_padded.tif'
    from skimage import io
    im = io.imread(pth)
    im_padded = crop_center(im, [50,50,50])
    from manage_files.read_save_files import save
    save(pth_save, im_padded)
    """
    im_path = "/home/eloy/Documents/stage_reconstruction_spfluo/real_data/high_res_images/results_unuseful_zone_suppressed/recons_shifted.tif"
    im_processed = "/home/eloy/Documents/stage_reconstruction_spfluo/real_data/high_res_images/results_unuseful_zone_suppressed/recons_shifted_thresh.tif"
    im = io.imread(im_path)
    im[im<0.1] = 0
    save(im_processed, im)


    from manage_files.paths import *
    from manage_files.apply_transformation_to_folder import transform_folder
    pth_raws = f'{PATH_REAL_DATA}/high_res_images/unuseful_zone_suppressed'
    pth_thrshold = f'{PATH_REAL_DATA}/high_res_images/unuseful_zone_suppressed_2'
    arg_threshold = {'thresh_val':0}
    #transform_folder(pth_raws, pth_thrshold, [threshold], [arg_threshold])

    pth_data = f'{PATH_REAL_DATA}/U-ExM/data/raw/c1_crop_60'
    pth_res = f'{PATH_REAL_DATA}/U-ExM/data/projected_ims'

    #transform_folder(pth_data, pth_res, [projection_z_axis], [{}])

    for fn in os.listdir(pth_res):
        new_fn = ''
        for st in fn:
            if st != '%':
                new_fn+=st
        os.rename(f'{pth_res}/{fn}', f'{pth_res}/{new_fn}')
