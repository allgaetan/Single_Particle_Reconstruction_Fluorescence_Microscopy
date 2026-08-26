import argparse
from manage_files.read_save_files import load_pickle
from reconstruction_with_dl.end_to_end_architecture_volume import train
from reconstruction_with_dl.data_set_views import ViewsRandomlyOrientedSimData
from data_generation.generate_data import heterogene_views_from_centriole
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pth1", dest="params_learn_setup_pth")
parser.add_argument("--pth2", dest="params_data_gen_pth")
args = parser.parse_args()

params_learn_setup_pth = getattr(args, "params_learn_setup_pth")
params_data_gen_pth = getattr(args, "params_data_gen_pth")

params_learn_setup = load_pickle(params_learn_setup_pth)
params_data_gen = load_pickle(params_data_gen_pth)

x = 100
list_l_channel_1 = np.linspace(30, 350, params_data_gen["nb_views"])
list_l_channel_2 = np.array(list(np.linspace(30,100, x)) + [100] * (params_data_gen["nb_views"] - x))


cs = [103.25, 50]
nb_channels = params_learn_setup["nb_channels"]
views, het_vals, rot_mats, rot_vecs, transvecs, gt_4d = heterogene_views_from_centriole(params_data_gen["pth_views"], params_data_gen, nb_channels,
                                                                                 [list_l_channel_1, list_l_channel_2], cs=cs)
file_names = ["" for _ in range(params_data_gen["nb_views"])]

data_set = ViewsRandomlyOrientedSimData(views, rot_mats, rot_vecs, transvecs, het_vals[0],
                                                    params_data_gen["size"], params_data_gen["nb_dim"], file_names)

train(data_set, params_data_gen, params_learn_setup, gt_4d)


