import numpy as np

from manage_files.apply_transformation_to_folder import transform_folder, apply_transformations_to_image
from manage_files.paths import *
from manage_files.read_save_files import save, make_dir
from common_image_processing_methods.barycenter import center_barycenter
from common_image_processing_methods.others import resize_to_given_size_of_pixels, crop_center, normalize, resize
from skimage import io


"""proprocessed data : 
1) resizes images so that pixel size is the same in x, y and z
2) center image barycenter at image center
3) crop images so that they have the same number of pixels in x, y and z
4) normalize images between 0 and 1"""

resize_z_ratio = 2.1
pixel_size_lateral_plane_before_rescaling = 1 # = 56 #1# 35 (UeXM)          #= 11.48 (high res images)    #= 56 (ISIM)
pixel_size_z_before_rescaling = resize_z_ratio # 41 (UeXM)           #= 15.54 (high res images) #150 (ISIM)
desired_pixel_size = 1 #= 35     # 70
crop_size = 80


path_raw_psf = "/data/eloy/TREx/Estimated_PSF_deconv.tif"
fold_preprocessed_psf = "/data/eloy/TREx"
"""
raw_psf = io.imread(path_raw_psf)
preprocessed_psf = resize_to_given_size_of_pixels(raw_psf, pixel_size_before=(pixel_size_z_before_rescaling, pixel_size_lateral_plane_before_rescaling, pixel_size_lateral_plane_before_rescaling),
                                                  pixel_size_after=(desired_pixel_size, desired_pixel_size, desired_pixel_size))
make_dir(fold_preprocessed_psf)
save(f"{fold_preprocessed_psf}/psf_preprocessed_deconv.tif", preprocessed_psf)

"""
folder_in = f'/data/eloy/TREx/deconv/picked/cropped/c1'
folder_out = f'/data/eloy/TREx/deconv/picked/cropped/c1_resized'
args_resize = {'pixel_size_before':(pixel_size_z_before_rescaling, pixel_size_lateral_plane_before_rescaling, pixel_size_lateral_plane_before_rescaling),
       'pixel_size_after':(desired_pixel_size, desired_pixel_size, desired_pixel_size)} # pixel_size_before : pixels sizes
# of the images in nm before rescaling (one pixel size for each dimension).  'pixel_size_after' : pixels sizes after rescaling
args_crop = {'size':(crop_size,crop_size,crop_size)}
args_normalize = {'min':0, 'max':1}

"""
transform_folder(folder_in, folder_out, [crop_center, center_barycenter],[args_crop,  {}])
         #[args_resize, {}, args_crop, args_normalize])
"""
"""
transform_folder(folder_in, folder_out, [resize, center_barycenter, crop_center, normalize],
         [{'desired_shape':(50,50,50)}, {}, args_crop, args_normalize])

"""

transform_folder(folder_in, folder_out, [resize_to_given_size_of_pixels, center_barycenter, crop_center, normalize],
                 [args_resize, {}, args_crop, args_normalize])

