import matplotlib.pyplot as plt
from manage_matplotlib.colors import gen_colors
from manage_matplotlib.graph_setup import set_up_graph
import numpy as np
from manage_files.read_save_files import *
from manage_files.paths import *



def plot_experiment_graph(x_values, y_values, xlabel, ylabel, title, labels, yrange=None, colors=None, fill_min_max=True, font_size=70):
    """x values : 1D array , values to plot in the x axis of size M
    y values : list of L array of size(L[i]) = (M, N_i).
    L --> number of graphs to plot
    M --> number of values in x axis
    N_i --> number of experiment per point for graph i
    labels : list of size L, labels[i] is the label to give to graph i
    colors : list of size L, colors[i] is the color of graph i"""
    if colors is None:
        colors = gen_colors(len(y_values))
    set_up_graph(SMALLER_SIZE=font_size)
    fig, ax = plt.subplots(figsize=(40,20))
    for i in range(len(y_values)):
        mean_vals = np.mean(y_values[i], axis=1)
        std_vals = np.std(y_values[i], axis=1)
        max_vals = np.max(y_values[i], axis=1)
        min_vals = np.min(y_values[i], axis=1)
        plt.errorbar(x_values, mean_vals, std_vals, linestyle='dotted', marker='o', c=colors[i], label=labels[i], linewidth=5)
        if fill_min_max:
            plt.fill_between(x_values, min_vals, max_vals, alpha=0.2)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    #plt.xlim(x_values[0], x_values[-1]+100)
    if yrange is not None:
        plt.ylim(yrange)
    plt.legend()
    plt.grid()
    plt.title(title)
    return ax, colors



def plot_graphs(x_values, graphs_to_plot, x_label, y_label, title, labels=None, colors=None, linewidth=1):
    if labels is None:
        labels = [None]*len(graphs_to_plot)
    if colors is None:
        colors = gen_colors(len(graphs_to_plot))
    for i in range(len(graphs_to_plot)):
        if labels[i] is not None:
            plt.plot(x_values, graphs_to_plot[i], label=labels[i], c=colors[i], linewidth=linewidth)
        else:
            plt.plot(x_values, graphs_to_plot[i], c=colors[i], linewidth=linewidth)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()



if __name__ == '__main__':
    gts = ["HIV-1-Vaccine_prep"]  # , "HIV-1-Vaccine_prep", "emd_0680", "recepteurs_AMPA"]
    sigma_gaussians = [0.2, 0.15, 0.12, 0.1, 0.08, 0.07, 0.05, 0.03]
    nb_tests_per_points = 10
    fold_gt = "../../ground_truths"
    for ground_truth_name in gts:
        props = np.arange(0.8, 4, 0.2)
        ssims = np.zeros((len(sigma_gaussians), len(props), nb_tests_per_points))
        fscs = np.zeros((len(sigma_gaussians), len(props), nb_tests_per_points))
        fold_root_0 = f'{PTH_LOCAL_RESULTS}/gmm_test_nb_gaussians/{ground_truth_name}/init_with_avg_of_views'
        for i, sig in enumerate(sigma_gaussians):
            fold_root = f'{fold_root_0}/sig_{sig}'
            make_dir(fold_root)
            gt = read_image(f'{fold_gt}/{ground_truth_name}.tif')
            ssims_sig = read_csv(f'{fold_root}/ssims.csv')
            fscs_sig = read_csv(f'{fold_root}/fscs.csv')
            ssims[i, :, :] = ssims_sig
            fscs[i, :, :] = fscs_sig

        labels = [f'sigma = {sig / 2}' for sig in sigma_gaussians]
        plot_experiment_graph(props, ssims, 'proportion of the optimal number of gaussians', 'ssim', '', labels)
        plt.savefig(f'{fold_root_0}/ssim_wrt_nb_gauss.png')
        plt.close()

        plot_experiment_graph(props, fscs, 'proportion of the optimal number of gaussians', 'Fourier Shell Correlation',
                              '', labels)
        plt.savefig(f'{fold_root_0}/fsc_wrt_nb_gauss.png')
        plt.close()









