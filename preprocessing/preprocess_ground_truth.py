import numpy as np
from common_image_processing_methods.others import resize, normalize
from common_image_processing_methods.barycenter import center_barycenter
from manage_files.read_save_files import save
from skimage import io

from common_image_processing_methods.otsu_thresholding import otsu_thresholding
from preprocessing.padding import pad_im_to_include_sphere_containing_object

"""preprocess ground_truth image
1) thresholds the image with otsu thresholding
2) normalizes 
3) centers the barycenter of object at image center
4) pad or crop the image so that the sphere containing the object is included in the image"""

# pth_raw_ground_truth = "../../ground_truths/Vcentriole.tif"
pth_raw_ground_truth = "../../ground_truths/recepteurs_AMPA.tif"
# pth_raw_ground_truth = "../../results/synthetic_data/recepteurs_AMPA/3D/ground_truth.tif"
output_location = "../../ground_truths/recepteurs_AMPA_prep2.tif"
size = 50 # size of the output image
sig_blur_otsu = 10 # standard deviation of the blurred applied in otsu thresholding method
im = io.imread(pth_raw_ground_truth)
im = resize(im, (size, size, size))
im = otsu_thresholding(im, sig_blur_otsu)
im = center_barycenter(im)
im = pad_im_to_include_sphere_containing_object(im, h=0)
im = resize(im, (size, size, size))
im[im<0] = 0
im = normalize(im)
im = np.pad(im, 5)
im = resize(im, (size, size, size))
save(output_location, im)

















