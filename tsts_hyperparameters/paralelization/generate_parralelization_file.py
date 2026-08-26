import os
from manage_files.read_save_files import make_dir

hyper_parameter_name = "dec_prop"

hyper_param_vals = {'dec_prop':[1.01, 1.02, 1.03, 1.04, 1.05, 1.07, 1.1, 1.2,1.3,1.5,1.7,2],
                  'snr':[0.1,0.5,1,2,5,10,20,30,50,70,100],
                  'sig_z':[1,3,5,7,10,12,15,17,20],
                  'nb_views':[60,80],
                  'N_rot':[5,10,12,15,17,20,22,25,30,35,40],
                  'coeff_kernel_axes':[5,10,20,30,40,50,60,70,80,90,100],
                  'coeff_kernel_rot':[1,5,10,20,30,40,50,60,70,80,90,100],
                    'lr':[0.01,0.05,0.07,0.08,0.09,0.1,0.12,0.13,0.14,0.15,0.16,0.18,0.2,0.25,0.5,1.0],
                    'N_sample':[50,100,150,200,250,300,400,500,600,700,800,1000,1200]}


#hyper_param_vals = {'dec_prop':[1.01,1.02,1.05,1.2]}
#hyper_param_vals = {'coeff_kernel_rot':[1,5,10,20]}
N_iter_max_associated_vals = [200,100,70,30]

if __name__ == '__main__':

    nb_tests_per_point = 10
    for hyper_parameter_name in hyper_param_vals.keys():
        ground_truth_name = "recepteurs_AMPA"
        fold_ground_truth = "../ground_truths"
        make_dir('parallelization_files')
        fn = f'parallelization_files/test_{hyper_parameter_name}_{ground_truth_name}_2'
        file = open(fn, 'w')
        for i,hyp_val in enumerate(hyper_param_vals[hyper_parameter_name]):
            for t in range(nb_tests_per_point):
                folder_results = f'../results/{ground_truth_name}/{hyper_parameter_name}_{hyp_val}/test_{t}'
                folder_views = f'{folder_results}/views'
                cmd = f'python parser.py --fold_ground_truth {fold_ground_truth} --ground_truth_name {ground_truth_name}.tif ' \
                        f'--folder_views {folder_views} --folder_results {folder_results} --{hyper_parameter_name} {hyp_val} --N_iter_max {N_iter_max_associated_vals[i]} --eps -5'
                file.write(f'{cmd} \n')

