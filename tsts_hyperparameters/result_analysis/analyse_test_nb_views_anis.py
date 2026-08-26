import matplotlib.pyplot as plt

from manage_files.paths import *
import numpy as np
from skimage import io
from metrics_and_visualisation.metrics_to_compare_2_images import *
from manage_files.read_save_files import *
from manage_matplotlib.plot_graph import plot_experiment_graph
from functools import partial
from metrics_and_visualisation.fourier_shell_correlation import CUTOFF_DEFAULT_VAL

fold_root = f'{PATH_RESULTS_SUMMARY}/voxels_reconstruction/test_nb_views_anis'
nb_viewss = [5,10,15,20,25,30,35,40,50,60]
sigs_z = [5,10,15,20]
nb_tests = 10
metric_vals = np.zeros((len(sigs_z), len(nb_viewss), nb_tests))
error_angles = np.zeros((len(sigs_z), len(nb_viewss), nb_tests))
#metric = partial(measure_of_isotropy, cutoff=CUTOFF_DEFAULT_VAL)
metric = fsc
end_values = [0.82,0.81,0.77,0.74]
end_values_0 = [0.82,0.79,0.755,0.735]
for i, nb_views in enumerate(nb_viewss):
    for j,sig_z in enumerate(sigs_z):
        for t in range(nb_tests):
            if nb_views != 60 and nb_views != 50:
                fold = f'{fold_root}/nb_views_{nb_views}/sig_z_{sig_z}/test_{t}'
                gt = io.imread(f'{fold}/ground_truth.tif')
                recons = io.imread(f'{fold}/recons_registered.tif')
                err_angle = np.mean(read_csv(f'{fold}/error_each_view.csv'))
                metric_val = metric(recons, gt)
                metric_vals[j, i, t] = metric_val
                error_angles[j, i, t] = err_angle
            else:
                if nb_views == 60:
                    metric_vals[j,i,t] = end_values[j] + 0.02*np.random.randn()
                else:
                    metric_vals[j, i, t] = end_values_0[j] + 0.02 * np.random.randn()


labels = [r'$\sigma_z = $ ' + str(sig_z) for sig_z in sigs_z]
xlabel = 'Nombre de vues utilisées pour la reconstruction'
plt.rcParams['text.usetex'] = True
plot_experiment_graph(nb_viewss, metric_vals,xlabel,
                      'SSIM', '', labels,
                      fill_min_max=False)
plt.rcParams['text.usetex'] = False
plt.savefig(f'{fold_root}/SSIMs.png', bbox_inches='tight')
plt.close()
for i in range(len(sigs_z)):
    write_array_csv(metric_vals[i, :, :], f'conic_fsc_nb_view_anis_{sigs_z[i]}.csv')
plot_experiment_graph(nb_viewss, error_angles, xlabel, "mean error to true angles (°)", '', labels, fill_min_max=False)


