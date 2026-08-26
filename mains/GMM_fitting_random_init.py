from learning_algorithms.GMM_coarse_to_fine import coarse_to_fine_gmm_optimization
from skimage import io
import numpy as np
from manage_files.read_save_files import make_dir

path = "../../results/real_data/unique_views/ex7/FOV_97_particle2_c1_raw.tif"
view = io.imread(path)
folder_results = "../../results/real_data/unique_views/ex7_result"
make_dir(folder_results)
cov_PSF = np.zeros((3,3))
true_rot_vecs = np.zeros((1,3))
true_trans_vecs = np.zeros((1,3))


coarse_to_fine_gmm_optimization(folder_results, np.array([view]), nb_gaussians_init=15, sigma_init=0.2, nb_gaussian_ratio=3, sigma_ratio=1.4, nb_steps=4,
                                    nb_dim=3, size=50, cov_PSF=cov_PSF, thershold_gaussians=0.1, true_rot_vecs=true_rot_vecs, true_trans_vecs=true_trans_vecs,
                                    eps=1, lr=5*10**-5, known_trans=True,known_axes=True, known_rot=True ,N_iter_maxs=100,
                                    init_unif_prop=None, dec_factor_prop=None, unif_prop_mins=None, N_axes=None, N_rot=None, M_axes=5000, M_rot=360,
                                    coeff_kernel_axes=None, coeff_kernel_rot=None)

