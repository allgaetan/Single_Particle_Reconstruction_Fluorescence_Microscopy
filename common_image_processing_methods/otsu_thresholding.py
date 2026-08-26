from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import copy as cp


def otsu_thresholding(image, sig_blur):
    blurred = gaussian_filter(image, sigma=sig_blur)
    t = threshold_otsu(blurred)
    im_thresh = cp.deepcopy(image)
    im_thresh[blurred < t] = 0
    return im_thresh
