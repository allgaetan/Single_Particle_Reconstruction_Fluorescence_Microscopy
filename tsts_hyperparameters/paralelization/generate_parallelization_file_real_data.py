from manage_files.read_save_files import make_dir



hyper_parameter_name = "dec_prop"
hyper_param_vals = {'coeff_kernel_axes':[50,60,70,80,90,100],
                  'coeff_kernel_rot':[1,2,3,4,5,6,7,8,9,10,12,15,20]}

nb_tests_per_point = 2
for hyper_parameter_name in hyper_param_vals.keys():
    make_dir('../parallelization_files/real_data')
    fn = f'../parallelization_files/real_data/test_{hyper_parameter_name}_{ground_truth_name}'
    file = open(fn, 'w')
    for hyp_val in hyper_param_vals[hyper_parameter_name]:
        for t in range(nb_tests_per_point):
            folder_results = f'../../results/data_marine/{hyper_parameter_name}_{hyp_val}/test_{t}'
            folder_views = f'{folder_results}/views' #a définir
            path_psf = None
            cmd = f'python parser.py --folder_views {folder_views} --folder_results {folder_results} --path_psf {path_psf} --{hyper_parameter_name} {hyp_val}'
            file.write(f'{cmd} \n')