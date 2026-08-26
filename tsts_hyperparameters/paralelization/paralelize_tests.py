import numpy as np
from manage_files.read_save_files import make_dir
from mains.synthetic_data.main_pixel_representation_synthetic_data import main_pixel_representation
from manage_files.read_save_files import write_array_csv, read_csv
import matplotlib.pyplot as plt
import multiprocessing as mp
from time import time

fold_gt = "../../ground_truths"
ground_truth_name = "recepteurs_AMPA"

folder_results = f"../../../results/{ground_truth_name}"
folder_views = f"../../../results/views/{ground_truth_name}"

output = mp.queues

processes = []
hyp_name = "lr"
hyp_vals = [0.01,0.05,0.07,0.1]
nb_tests_per_val = 4
t = time()
for p in range(len(hyp_vals)):
    for t in range(nb_tests_per_val):
        folder_results = f'{folder_results}_{hyp_name}_{hyp_vals[p]}_{t}'
        folder_views = f'{folder_views}_{hyp_name}_{hyp_vals[p]}_{t}'
        process = mp.Process(target=main_pixel_representation, args=(fold_gt, f'{ground_truth_name}.tif', folder_results, folder_views)
                         ,kwargs={hyp_name:hyp_vals[p]})
        processes.append(process)

for p in processes:
    p.start()

for p in processes:
    p.join()

print("temps d'execution", time()-t)







