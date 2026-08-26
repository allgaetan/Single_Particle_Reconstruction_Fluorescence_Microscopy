from stage_gaetan.paths import PROJECT_PATH
from common_image_processing_methods.others import *
from data_generation.generate_data import heterogene_views_from_centriole, heterogene_views_from_centriole_2_degrees_of_freedom, generate_and_save_data

def generate(params_data_gen, params_learn_setup, gt_name="", dl=1):
    """
    Génère les vues à partir d'un ground truth
    Actuellement ne peut que générer des données homogènes à 1 channel, 
    hétérogène (centriole) avec 1 degré de liberté et 1 ou 2 channels, 
    et hétérogène (centriole) avec 2 degrés de liberté et 1 channel
    - params_data_gen: paramètres de génération de données
    - params_learn_setup: paramètres de l'apprentissage
    - gt_name: nom du ground truth à utiliser pour la génération
    - dl: nombre de degrés de liberté pour l'hétérogénéité (1 ou 2)
    """
    if not params_learn_setup["heterogeneity"]: # homogène
        gt_path = f"{PROJECT_PATH}/ground_truths/"
        save_fold = f"{params_data_gen['pth_views']}"
        generate_and_save_data(save_fold, gt_path, gt_name + ".tif", params_data_gen)
    else:
        """
        if dl==1: # hétérogène 1 dl 1/2 channels
            save_fold = f'{PATH_PROJECT_FOLDER}/heterogen/views/1dl'
            list_l_channel_1 = np.linspace(30, 350, 250)
            list_l_channel_2 = np.concatenate((np.linspace(30, 100, 100), np.linspace(100, 100, 150)))
            cs = [103.25, 50]
            heterogene_views_from_centriole(
                save_fold=save_fold,
                param_data_gen=params_data_gen, 
                nb_channels=params_learn_setup["nb_channels"],
                cs=cs,
                het_vals_all_channels = [list_l_channel_1, list_l_channel_2], 
                zero_rotation=False
            )
        if dl==2: # hétérogène 2 dl 1 channel
            save_fold = f'{PATH_PROJECT_FOLDER}/heterogen/views/2dl'
            het_vals_lenght = np.linspace(30,350,50)
            het_vals_radius = np.linspace(50,110,30)
            heterogene_views_from_centriole_2_degrees_of_freedom(
                save_fold=save_fold, 
                param_data_gen=params_data_gen, 
                het_vals_lenght=het_vals_lenght, 
                het_vals_radius=het_vals_radius
            )
        """
        pass
    return save_fold