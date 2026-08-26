from reconstruction_with_dl.end_to_end_architecture_volume import train, End_to_end_architecture_volume
from manage_files.read_save_files import read_image, read_images_in_folder
from reconstruction_with_dl.data_set_views import ViewsRandomlyOriented
from manage_files.read_save_files import write_array_csv
from manage_files.paths import PATH_REAL_DATA
import numpy as np
from reconstruction_with_dl.test_params.default_params import params_learn_setup, params_data_gen

multichannel = True
params_data_gen["sig_z"] = 3

params_learn_setup["nb_epochs"] = 4000
params_learn_setup["x"] = 50
params_learn_setup["nb_dim_het"] = 7
params_learn_setup["nb_epochs_each_phases_ACE_Het"] = None
params_learn_setup["knwon_trans"] = False


if not multichannel:
    pth_real_data = f"{PATH_REAL_DATA}/Assembly_cropped_normalized"
    save_fold = f"{PATH_REAL_DATA}/results_assembly/week_25_mars/test_reg_trans"
    params_learn_setup['nb_channels'] = 1
else:
    #pth_real_data = f'{PATH_REAL_DATA}/SAS6/picking/deconv_cropped_proto/side_views'
    pth_real_data = f'{PATH_REAL_DATA}/SAS6/picking/deconv_cropped_proto/all_views'
    save_fold = (f"{PATH_REAL_DATA}/SAS6/results_all_views/test_impose_same_radius_same_width")
    params_learn_setup['nb_channels'] = 2
    params_learn_setup["impose_cylinder"] = True

params_learn_setup["save_fold"] = save_fold
views, fns = read_images_in_folder(pth_real_data, alphabetic_order=False, multichannel=multichannel)

sum_c1  = np.sum(views[:,0,:,:,:])
sum_c2  = np.sum(views[:,1,:,:,:])
params_learn_setup["coeff_channel_impose_cyl"] = sum_c1/sum_c2
params_learn_setup["coeff_regul_pos"] = 1

nb_views = len(views)
if not multichannel:
    views = np.expand_dims(views, axis=1)
params_data_gen["size"] = views[0].shape[1]
data_set = ViewsRandomlyOriented(views, params_data_gen["size"], 3, fns)
train(data_set, params_data_gen, params_learn_setup)
