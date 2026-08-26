from manage_files.read_save_files import make_dir
nb_tests_per_point = 30

fold_ground_truth = "../ground_truths"
make_dir('parallelization_files')
fn = '../parallelization_files/test_ground_truths'
file = open(fn, 'w')

for gt_name in ["recepteurs_AMPA", "HIV-1-Vaccine_prep", "clathrine", "emd_0680", "Vcentriole_prep"]:
    for t in range(nb_tests_per_point):
        folder_results = f'../results/test_gt/{gt_name}/test_{t}'
        folder_views = f'{folder_results}/views'
        cmd = f'python parser.py --fold_ground_truth {fold_ground_truth} --ground_truth_name {gt_name}.tif ' \
                    f'--folder_views {folder_views} --folder_results {folder_results}'
        file.write(f'{cmd} \n')