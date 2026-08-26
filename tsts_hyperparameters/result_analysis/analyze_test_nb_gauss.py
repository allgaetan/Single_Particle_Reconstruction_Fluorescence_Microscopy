from manage_files.paths import *
from manage_files.read_save_files import *
gts = ["clathrine", "HIV-1-Vaccine_prep", "emd_0680", "recepteurs_AMPA"]
sigma_gaussians = [0.2, 0.15, 0.12, 0.1, 0.08, 0.07, 0.05, 0.03]
nb_gaussians = [2,5,10,15,20,25,30,40,50,75,100,125,150,175,200,250]

for gt in gts:
    fscs_all = []
    proportion_opt_gauss = []
    for sig in sigma_gaussians:
        fold = f'{PTH_LOCAL_RESULTS}/gmm_test_nb_gaussians_2/{gt}/init_with_avg_of_views'
        fscs = read_csv(f'{fold}/fscs.csv')
        opt_nb_gauss = read_csv(f'{fold}/opt_nb_gaussians.csv')
        fscs_all.append(fscs)
        proportion_opt_gauss
