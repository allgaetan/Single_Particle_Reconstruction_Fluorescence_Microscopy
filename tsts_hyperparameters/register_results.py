from skimage import io
from common_image_processing_methods.registration import registration_exhaustive_search
from tsts_hyperparameters.paralelization.generate_parralelization_file import hyper_param_vals
from skimage.metrics import structural_similarity as ssim
from manage_files.read_save_files import write_array_csv
import numpy as np



"""
hyp_name = "coeff_kernel_axes"
fold_root = f"/home/eloy/Documents/stage_reconstruction_spfluo/results_hpc/recepteurs_AMPA/{hyp_name}"
hyp_vals = hyper_param_vals[hyp_name]
nb_tests = 5



for hyp_val in hyp_vals:
    print('hyp_val', hyp_val)
    for t in range(nb_tests):
        print('test', t)
        fold = f'{fold_root}/{hyp_name}_{hyp_val}/test_{t}'
        recons = io.imread(f'{fold}/recons.tif')
        ground_truth = io.imread(f'{fold}/ground_truth.tif')
        registration_exhaustive_search(f'{fold}/ground_truth.tif', f'{fold}/recons.tif', fold, 'recons_registered', 3,
                                       sample_per_axis=15)
        recons_registered = io.imread(f'{fold}/recons_registered.tif')
        gt = io.imread(f'{fold}/ground_truth.tif')
        sm = ssim(gt, recons_registered)
        write_array_csv(np.array([[sm]]), f'{fold}/ssim.csv')

"""
nb_views_list = [5,10,15,20,25,30,35,40,45,50,70]
#nb_views_list = [80]
fold_root = "/home/eloy/Documents/stage_reconstruction_spfluo/results_scipion/tomographic_reconstruction"
gt_names = ["emd_0680", "recepteurs_AMPA"]
for gt_name in gt_names:
    for nb_view in nb_views_list:
        gt_path = f"/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{gt_name}.tif"
        for t in range(1,11):
            pth = f'{fold_root}/{gt_name}/nb_views_{nb_view}/vol{t}.tif'
            im = io.imread(pth)
            registration_exhaustive_search(gt_path, pth, f'{fold_root}/{gt_name}/nb_views_{nb_view}', f'vol{t}_registered', 3,
                                           sample_per_axis=30)













