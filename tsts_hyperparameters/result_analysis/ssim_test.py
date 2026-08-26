import numpy as np

from manage_files.paths import *
from skimage import io
from metrics_and_visualisation.metrics_to_compare_2_images import *

def cov_xy(X, Y):
    return np.mean((X - np.mean(X))*(Y-np.mean(Y)))

def s(X,Y):
    return cov_xy(X, Y)/(np.std(X)*np.std(Y))

part = 'recepteurs_AMPA'
pth = f'{PATH_PROJECT_FOLDER}/view_gt_recons/{part}'

view = io.imread(f'{pth}/SVD.tif')

gt = io.imread(f'{pth}/ground_truth.tif')
recons = io.imread(f'{pth}/recons_registered.tif')



print('mean view', np.mean(view))
print('mean recons', np.mean(recons))
print('mean gt', np.mean(gt))

print('std view', np.std(view))
print('std recons', np.std(recons))
print('std gt', np.std(gt))

print('s view', s(gt, view))
print('s recons', s(recons, gt))

ssim_recons = ssim(gt, recons)
ssim_view = ssim(gt, view, gaussian_weights=True, sigma=4)

print('ssim recons', ssim_recons)
print('ssim view', ssim_view)





