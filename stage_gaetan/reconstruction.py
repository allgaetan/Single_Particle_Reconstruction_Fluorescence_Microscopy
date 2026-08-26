from common_image_processing_methods.others import *
from reconstruction_with_dl.data_set_views import ViewsRandomlyOrientedSimData
from reconstruction_with_dl.end_to_end_architecture_volume import train, read_heterogene_views_synth_data, read_homogene_views_synth_data, train_with_hps

def reconstruct(params_data_gen, params_learn_setup):
    """
    Lance la reconstruction à partir des vues générées
    - params_data_gen: paramètres de génération de données
    - params_learn_setup: paramètres de l'apprentissage
    """
    if not params_learn_setup["heterogeneity"]: # homogène
        views, rot_mats, rot_vecs, transvecs, dilatation_vals, file_names = read_homogene_views_synth_data(params_learn_setup, params_data_gen, True, False)
    else: # hétérogène
        pass
        """
        views, rot_mats, rot_vecs, transvecs, dilatation_vals, file_names = read_heterogene_views_synth_data(params_learn_setup, params_data_gen, True, params_learn_setup["nb_dim_het"]==2)
        if params_learn_setup["nb_channels"]==2:
            params_learn_setup["coeff_channel_impose_cyl"] = np.sum(views[:,0,:,:,:])/np.sum(views[:,1,:,:,:])  # relative weight of channel 2 wrt to channel 1 in the loss when we impose the tore shape
        """
    data_set = ViewsRandomlyOrientedSimData(
        views, 
        rot_mats, 
        rot_vecs, 
        transvecs,
        dilatation_vals, 
        params_data_gen["size"], 
        params_data_gen["nb_dim"], 
        file_names)
    print('start train')

    if params_learn_setup["HPS"]:
        train_with_hps(data_set, params_data_gen, params_learn_setup, None, views) # Entrainement de Fluofire avec HPS
    else:
        train(data_set, params_data_gen, params_learn_setup, None, views) # Entrainement classique de Fluofire
