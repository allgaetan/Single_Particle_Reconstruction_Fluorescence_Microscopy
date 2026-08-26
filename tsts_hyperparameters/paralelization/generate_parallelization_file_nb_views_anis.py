from manage_files.read_save_files import make_dir


ground_truth_name = "recepteurs_AMPA"
fold_ground_truth = "../ground_truths"
make_dir('parallelization_files')
fn = f'../parallelization_files/test_nb_views_anis'
nb_tests_per_point = 10
file = open(fn, 'w')

nb_viewss = [2,5,10,20,30,40,50,60,80]
sigs_z = [5,10,15,20]
for nb_views in nb_viewss:
    for sig_z in sigs_z:
        for t in range(nb_tests_per_point):
            folder_results = f'../results/{ground_truth_name}/nb_views_{nb_views}/sig_z_{sig_z}/test_{t}'
            folder_views = f'{folder_results}/views'
            cmd = f'python parser.py --fold_ground_truth {fold_ground_truth} --ground_truth_name {ground_truth_name}.tif ' \
                            f'--folder_views {folder_views} --folder_results {folder_results} --nb_views {nb_views} --sig_z {sig_z}'
            file.write(f'{cmd} \n')

