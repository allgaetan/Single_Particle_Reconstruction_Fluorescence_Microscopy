import numpy as np

from manage_files.read_save_files import read_csv
import matplotlib.pyplot as plt
from tsts_hyperparameters.paralelization.generate_parralelization_file import hyper_param_vals
from skimage import io
from common_image_processing_methods.others import normalize
from metrics_and_visualisation.metrics_to_compare_2_images import *
import os
from manage_matplotlib.plot_graph import plot_experiment_graph
from manage_files.paths import *


def fill_zero_values(arr):
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i,j] == 0:
                arr[i,j] = np.mean(arr[i][arr[i]!=0])


def fill_zero_values_2(arr):
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            for k in range(len(arr[0,0])):
                if arr[i,j,k] == 0:
                    arr[i,j,k] = np.mean(arr[i][arr[i]!=0])


hyper_par_name = "snr"
gt_name = "recepteurs_AMPA"

#fold = f'{PATH_RESULTS_SUMMARY}/gmm_views_test_nb_gauss/{gt_name}'
fold = f'{PATH_RESULTS_SUMMARY}/test_snr_unlnown_poses'
#fold = f"{PATH_RESULTS_SUMMARY}/voxels_reconstruction/{hyper_par_name}"
# vals_to_test = hyper_param_vals[hyper_par_name]
#vals_to_test = hyper_param_vals[hyper_par_name]
#vals_to_test = [5,10,20,30,40,50,60,70,80,90,100]
vals_to_test = [0.01, 0.02 , 0.05, 0.07,0.1,0.2,0.5,0.7,1]
nb_tests_per_points = [10]*len(vals_to_test)
# nb_tests_per_points[-1] = 3

nb_views = 10
metrics = np.zeros((len(vals_to_test), nb_tests_per_points[1]))
errors = np.zeros((len(vals_to_test), nb_tests_per_points[1], nb_views))

gt_path = f"/home/eloy/Documents/stage_reconstruction_spfluo/ground_truths/{gt_name}.tif"
# plus_text = ''
plus_text = '_gradient'
gt = io.imread(gt_path)
fscs_res = [[] for _ in range(len(vals_to_test))]
T = nb_tests_per_points[1]
metric = fsc
"""read volumes reconstructed by my methods and stores the metrics, the error to true angles 
and the fourier shell correlation"""
# fscs_res = np.zeros((len(nb_tests_per_points), size_fsc_res))

for i in range(len(vals_to_test)):
    for t in range(nb_tests_per_points[i]):
        fold_results = f'{fold}/{hyper_par_name}_{vals_to_test[i]}/test_{t}'
        recons_path = f'{fold_results}/recons_registered{plus_text}.tif'
        recons = io.imread(recons_path)
        recons = normalize(recons)
        #recons[recons<0.05] = 0
        metric_val = metric(recons, normalize(gt)) #, plot_path=f'{fold_results}/fsc_curve.png')
        #metric_val = read_csv(f'{fold_results}/ssims.csv')[-1]
        metrics[i, t] = metric_val
        """
        for t2 in range(t+1, nb_tests_per_points[i]):
            recons2 = io.imread(f'{fold}/{hyper_par_name}_{vals_to_test[i]}/test_{t2}/recons_registered.tif')
            fsc = calc_fsc(recons2, recons, 1)
            fsc_res = fsc2res(fsc)
            fscs_res[i].append(fsc_res)
        """
        
print('metrics', metrics)
fscs_res = np.array(fscs_res)
vals_to_test = np.array(vals_to_test)
fill_zero_values_2(errors)
fill_zero_values(metrics)


"""reads the results of the single view deconvolution method"""

#xlabel = hyper_par_name
#ylabel = metric.__name__
#xlabel = r'$N_dN_{\psi}$'
#xlabel = r'$\beta_{\psi}$'
xlabel = 'Rapport signal à bruit'
#xlabel = r'$\alpha_r$'
ylabel = "SSIM" if metric.__name__ == "structural_similarity" else "FSC"
yrange = (0.6, 0.95) if metric.__name__ == "structural_similarity" else (0.05,0.4)
X = vals_to_test
plt.rcParams['text.usetex'] = True
ax, _ = plot_experiment_graph(X, [metrics], xlabel, ylabel, '',[None], yrange)
plt.xscale('log')
plt.rcParams['text.usetex'] = False

"""
ax2 = ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(X, X*0.075,color="green", linewidth=5)
ax2.set_ylabel("Temps de calcul par itération (s)", color='green')
"""

plt.savefig(f'{fold}/graph_{metric.__name__}{plus_text}.png')
plt.close()



# time per energie evaluation vol 50*50*50 : 0.053s


#plot_experiment_graph(vals_to_test, [metrics_scipion], hyper_par_name, metric.__name__, '',
        #                   ['single view deconvolution', 'our method'])











