from volume_representation.gaussian_mixture_representation.GMM_grid_evaluation import gaussian_mixture_isotrop_identical_gaussians, make_grid
from manage_files.read_save_files import read_csv, save
from common_image_processing_methods.others import normalize
import numpy as np
from rotation_translation import translation, rotation, get_3d_rotation_matrix

"generates a 3d image from point cloud coordinates"

size = 100
nb_dim = 3
grid = make_grid(size, nb_dim)
point_cloud = read_csv("sample_centriole_point_cloud.csv", first_col=0)

point_cloud = normalize(point_cloud, min=-0.9, max=0.9)

coeffs = np.array([1]*len(point_cloud))

gen_im = gaussian_mixture_isotrop_identical_gaussians(grid, coeffs, point_cloud, 0.02, point_cloud.shape[1], 3)
gen_im = translation(gen_im, [0, 10, -5])
rot_mat = get_3d_rotation_matrix([0,90,0], convention='xyz')
gen_im, _ = rotation(gen_im, rot_mat)
gen_im = normalize(gen_im)
save('synth_centriole.tif', gen_im)
