import numpy as np
from  volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import make_grid, nd_gaussian
from manage_files.read_save_files import save

size = 30
pixel_size_xy = 56
pixel_size_z = 150
fwhm_psf_xy = 50 #200 # width of PSF in xy (nm) # = 50
fwhm_psf_z = 210 #800 # width of PSF in z (nm) # = 210

sigma_PSF_pixels_xy = fwhm_psf_xy/(np.sqrt(np.log(2))* pixel_size_xy)
sigma_PSF_pixels_z = fwhm_psf_z / (np.sqrt(np.log(2))*pixel_size_z)
cov_PSF = np.array([[sigma_PSF_pixels_z**2, 0, 0],
                    [0, sigma_PSF_pixels_xy**2, 0],
                    [0, 0, sigma_PSF_pixels_xy**2]])

grid_step = 2/(size-1)
cov_PSF = grid_step ** 2 * cov_PSF
grid = make_grid(size, 3)
PSF = nd_gaussian(grid, np.zeros(3), cov_PSF, 3)
PSF /= np.sum(PSF)
save('PSF.tif', PSF)