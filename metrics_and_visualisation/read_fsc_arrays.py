import matplotlib.pyplot as plt
from fourier_shell_correlation import CUTOFF_DEFAULT_VAL
from common_image_processing_methods.rotation_translation import discretize_sphere_uniformly
from manage_files.read_save_files import read_csv, save_figure
from metrics_and_visualisation.fourier_shell_correlation import plot_resolution_map, find_cutoffs_conical_fsc
import numpy as np

pth_root = "/home/eloy/Documents/stage_reconstruction_spfluo/reports/image_to_include_in_reports/conical_fsc_abs_dot"
for part_name in ["recepteurs_AMPA", "clathrine", "HIV-1-Vaccine_prep", "emd_0680", "Vcentriole_prep"]:
    for coeff_kernel_axes in [1,2,3,7,10]:
        pth_part = f'{pth_root}/{part_name}'
        in_folds = [f'{pth_part}/conical_fsc_array_2',
                    f'{pth_part}/recons/conical_fsc_array_2',
                    f'{pth_part}/spartran/conical_fsc_array_2']
        in_names = [f'cFSC_map_view_gt.pngcoeff_{coeff_kernel_axes}.csv', f'cFSC_map.pngcoeff_{coeff_kernel_axes}.csv', f'cFSC_map.pngcoeff_{coeff_kernel_axes}.csv']
        print(len(in_names))
        print(len(in_folds))

        out_folds = [f'{pth_part}', f'{pth_part}/recons', f'{pth_part}/spartran']
        out_names = [f'cFSC_map_view_gt.pngcoeff_{coeff_kernel_axes}.png', f'cFSC_map.pngcoeff_{coeff_kernel_axes}.png',
                     f'cFSC_map.pngcoeff_{coeff_kernel_axes}.png']
        min_cut_of = 0.5
        max_cut_off = 0
        conical_fscs = []
        nb_sectors = 500
        points_to_add = [(0,0), (0,180), (360,0), (360,180), (0,90), (360,90), (180,180), (180,0), (180, 90)]
        thetas, phis, _ = discretize_sphere_uniformly(nb_sectors - len(points_to_add))
        thetas = list(thetas)
        phis = list(phis)
        for i in range(len(points_to_add)):
            thetas.append(points_to_add[i][0])
            phis.append(points_to_add[i][1])
        radiuses = np.linspace(0, 0.5, 30)
        for j in range(len(in_names)):
            conical_fsc = read_csv(f'{in_folds[j]}/{in_names[j]}')
            conical_fscs.append(conical_fsc)
            radiuses_cut_off = find_cutoffs_conical_fsc(radiuses, conical_fsc, CUTOFF_DEFAULT_VAL)
            if np.max(radiuses_cut_off) >= max_cut_off:
                max_cut_off = np.max(radiuses_cut_off)
            if np.min(radiuses_cut_off) <= min_cut_of:
                min_cut_of = np.min(radiuses_cut_off)


        for j in range(len(conical_fscs)):

            radiuses_cut_off = plot_resolution_map(radiuses, thetas, phis, conical_fscs[j], vmin=min_cut_of, vmax=max_cut_off, label='resolution (1/pixel)')
            save_figure(out_folds[j], f'{out_names[j]}')
            plt.close()

