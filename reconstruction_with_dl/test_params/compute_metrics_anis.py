import numpy as np
from reconstruction_with_dl.test_params.compute_metricssnr import *
from manage_matplotlib.plot_graph import plot_experiment_graph
from manage_files.read_save_files import read_csv

comp = False
sigs_z = [1,3,5,10,15]
nb_viewss = [5,20,50,100,200,300,500]
#nb_viewss = [300, 500]
#sigs_z = [1,3]
#nb_viewss = [5,10]
pth_gt = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/2_channels'
pth_results = f'{PATH_PROJECT_FOLDER}/results_deep_learning/centriole/results_hpc/test_anis/recons_results'
nb_channels = 1
if not comp:
    gt = read_resize_gt(pth_gt, nb_channels)
else:
    gt = None
nb_tests = 5

fscs = np.zeros((len(sigs_z), len(nb_viewss), nb_tests))
ssims = np.zeros((len(sigs_z), len(nb_viewss), nb_tests))
corrs = np.zeros((len(sigs_z), len(nb_viewss), nb_tests))

for i,sig_z in enumerate(sigs_z):
    print('sig z', sig_z)
    for j,nb_views in enumerate(nb_viewss):
        ep = int(2000 * 250 / nb_views - 2) if nb_views < 100 else 2999
        for t in range(nb_tests):
            pth_im = f'{pth_results}/nb_views_{nb_views}__sig_z_{sig_z}/test_{t}/ep_{ep}'
            fsc_avg, ssim_avg, val_corrcoeff = zbeul(pth_im, nb_channels, gt, nb_views, compute_only_corrcoeff=comp)
            fscs[i,j,t] = fsc_avg
            ssims[i,j,t] = ssim_avg
            corrs[i,j,t] = val_corrcoeff

xlabel = 'Nombre de vues utilisées pour la reconstruction'
labels = [r'$\sigma_z = $ ' + str(sig_z) for sig_z in sigs_z]
plt.rcParams['text.usetex'] = True

if not comp:
    plot_experiment_graph(nb_viewss, fscs, xlabel, "FSC", "", labels, fill_min_max=False)
    plt.xticks(nb_viewss)
    #plt.xscale('log')
    plt.savefig(f'{pth_results}/fscs.png')
    plt.close()

    plot_experiment_graph(nb_viewss, ssims, xlabel, "SSIM", "", labels, fill_min_max=False)
    plt.xticks(nb_viewss)
    #plt.xscale('log')
    plt.savefig(f'{pth_results}/ssims.png')
    plt.close()

plot_experiment_graph(nb_viewss, corrs, xlabel, "Corrélation hétérogénéité réelle/estimée", "", labels, fill_min_max=False)
plt.xticks(nb_viewss)
plt.savefig(f'{pth_results}/corrs.png')
plt.close()

plt.rcParams['text.usetex'] = False

for i,sig_z in enumerate(sigs_z):
    if not comp:
        write_array_csv(ssims[i], f'{pth_results}/ssims_sig_z_{sig_z}.csv')
        write_array_csv(fscs[i], f'{pth_results}/fscs_{sig_z}.csv')
    write_array_csv(corrs[i], f'{pth_results}/corrs_{sigs_z}.csv')


