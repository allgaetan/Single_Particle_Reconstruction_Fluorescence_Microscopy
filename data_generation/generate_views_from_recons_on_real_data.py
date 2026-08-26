import numpy as np

from manage_files.paths import *
from manage_files.read_save_files import *
from data_generation.generate_data import generate_one_view

fold_recons = f'{PATH_REAL_DATA}/U-ExM/data/raw/results/c1_crop_80_4_test'
path_recons = f'{fold_recons}/recons.tif'
path_angles = f'{fold_recons}/intermediar_results/estimated_rot_vecs_epoch_22.csv'

angles_founds = read_csv(path_angles)
views_orders = ['C1-24th_0.7%FA_1% AA_016\uf022Series016.tif', 'C1-24th_0.7%FA_1% AA_013\uf022Series014.tif', 'C1-24th_0.7%FA_1% AAbis_010\uf022Series011.tif', 'C1-24th_0.7%FA_1% AAbis_011\uf022Series012.tif', 'C1-24th_0.7%FA_1%AAbis_000\uf022Series001.tif', 'C1-24th_0.7%FA_1% AAbis_007\uf022Series008.tif', 'C1-24th_0.7%FA_1% AAbis_001\uf022Series002.tif', 'C1-24th_0.7%FA_1% AAbis_006\uf022Series007.tif', 'C1-24th_0.7%FA_1% AA_011\uf022Series012.tif', 'C1-24th_0.7%FA_1% AAbis_003\uf022Series004.tif', 'C1-24th_0.7%FA_1% AAbis_004\uf022Series005.tif', 'C1-24th_0.7%FA_1% AAbis_012\uf022Series013.tif', 'C1-24th_0.7%FA_1% AAbis_005\uf022Series006.tif', 'C1-REF_24th_0.7%FA_1% AAbis_009\uf022Series010.tif']
recons = io.imread(path_recons)
path_psf = f"{PATH_REAL_DATA}/U-ExM/data/raw/psfc1_preprocessed.tif"
folder_res = '/home/eloy/views_generated_from_recons_with_angles_found'
make_dir(folder_res)
for i in range(len(views_orders)):
    generated_view = generate_one_view(recons, 5, 1, angles_founds[i, :], np.zeros(3), 100)
    save(f'{folder_res}/{views_orders[i]}', generated_view)