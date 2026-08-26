import numpy as np
import pickle


params_learn_setup = {"save_fold":'.', # path of folder in which results are saved
            "nb_channels" :1, # number of channels of views (= number of channel of reconstructed object)
            "sym_loss": False, # if true, the symmetry loss is used in the learning process (see file "losses")
            "nb_epochs_each_phases_ACE_Het": None, #[20, 20], # can take either a None value or a 2 elements list [n,m].
                                # if None, we use only image to image learning phases. If in the form [n,m],
                                # alterates between n epochs of image to image
                                # learning phases and m epochs of pose to pose learning phases
            "heterogeneity": False, # if true, estimate a conformation parameter
            "nb_dim_het" :1, #7 if impose cylinder # dimension of conformational parameter, used iff heterogeneity is true
              "regul_volume_param":10**-4, # coefficient of regularization on volume
              "regul_heterogeneity_param":10**-4, #coefficient of regularization on heterogeneity parameters
            "impose_symmetry" : False, # if true impose a nine fold symmetry to the reconstruction
            "known_rot":False, # if true, rotation are not estimated and are set to true value
            "known_trans" :False, # if true, translation are not estimated and are set to true value
            "nb_epochs":2000, # number of epochs (one epochs = one passage through each view)
            "bs":1, #bathc size
            "x" : 100, # results are saved every x epochs
            "device":0, # device on which deep kearning model is trained
            "init" :'random', # specify how the initial volume is inizialized (see function init_volume in end_to_end_architecture_volume.py file
                             # for details.
            "impose_cylinder" :False, # if true, we impose a form of tore to the recontruction and try to estimate the parameters of it.
                                      # otherwise a siren representation is used
            "coeff_channel_impose_cyl":9, # specifiy the relative weight of channel 2 in the loss when we use the cylinder model.
                                         # usually, this must be set to the average ratio of intensity of the views of channel 1 on the views of channel 2
            "cylinder_maxmin_parameters": {"min_lenght":0.04, "max_lenght":1.8, "min_radius":0.1, "max_radius":0.8,
                                           "min_width":0.03, "max_width":0.6, "min_pos":0, "max_pos":0.9}, # donut parameter min and max values (imposed thanks to a sigmoid function)
            "encoder_name":'holly', #options : 'holly', 'vgg' (see top of pose_net.py file)
            "rot_representation":'6d', #options : 'axis_angle', 'euler', 'quaternion', '6d' (see top of pose_net.py file)
            "rot_args":{}, # arguments of rotation representation (for example "ZXZ") when we use the euler representation
            "batch_norm_rot":False, # if True, use bathc norm on the rotations
            "batch_norm_trans":False,
             "coeff_trans":0.01, #the part of the output of decoders that corresponds to tranlations parametres is multiplied by this coeeficient
                                # otherwise, the translation are to high at the beginning and the model falls in teh local minima
             "coeff_reg_trans_ratio":1, # a regularisation is used on the translation parameters. The weight of the regularization coefficient
                                        # is divided by this parameter at each epoch
             "init_coeff_reg_trans":1, # initial value of regularization coefficient on translation
             "use_reg_trans":True, # iff True, use regularization on translation
             "init_gain":1, # initialization gain for the eoncoder (see file pose_net.py)
             "omega":30, # pulastion of siren network (see siren.py file)
             "nb_hidden_layers":3, # number of hidden layers of siren network
             "nb_hidden_features":256, # number of feature in each hidden layer of siren
              "lr_pose_net":10**-4, #learning rate of pose net updates
              "lr_siren":5*10**-5, #learning rate of siren updates
              "lr_pos_param":5*10**-3,
              "relu":False, # use relu activation function instead of sinusoidal function on the siren
              "loss_type":'l2',  # option "l2" --> erreur qudratique moyenne option "l1" --> erreur moyenne
              "use_sepctral_norm":False, # use of spectra norm for convolutional layer of pose net
              "freeze_encoder":None, # number of epoch the encoder will be frozen at the begining of learning. If None, the encoder is not freezed at all
              "start_ep_learn_het":0} # epoch from which the heterogeneity param starts to be learnt

params_data_gen = {"pth_views":None, #path of the input views
                    "size" : 50, # size of one side of the volume
                    "nb_dim" : 3, # dimension of the volume
                    "sig_z" :1, # standard deviation of psf along z axis
                    "sig_xy" : 1, # standard deviation of psf on xy plane
                    "nb_views" :250, # number of generated views
                     "snr":1000, # signal to noise ratio
                     "sigma_trans_ker":0, # translation are generated according to a normal distribution centered in 0 and with standard deviation sigma_trans_ker
                     "partial_labelling":False, # use or not of partial labelling
                      "partial_labelling_args":{}, #argument of partial labelling (see function simulate_partial_labelling in simulate_partial_labelling.py file in folder data_generation)
                      "projection":False, # if true a projection is performed before the convolutopn (to simulate cryo-ME images instead of fluorescence images)
                      "rot_vecs":None, # if None generate views with rot_vecs selected accorded to an uniform distribuition. Otherwise, specifiy a list of lenght
                                        #nb_views containing the rotation vectors you want to use to generate all the views
                      "convention":"ZXZ", # convention of rotation generation
                      "order":3, #order of interpolation used for rotation
                      "rotation_max":None, # if not None, take the form [a,b,c] a, b and c being respectively the maximum value of rotation along of first second and
                                            # third element of rot_vec
                      "no_psf":False, #if true use non psf in generated data
                      "psf":None, # if not none use directly this psf to generate the views instead of a psf generated from the two standard deviations sig_z and sig_xy
                      "max_dil_val":None}








class ParametersLearningSetup:
    def __init__(self, heterogeneity=False, nb_dim_het=1, encoder_name='holly', rot_representation='6d', use_sym_loss=False,
                device=0, nb_epochs=100, bs=8, init='random', omega=30, coeff_trans=0.05, init_gain=1, batch_norm_rot=False, batch_norm_trans=False,
                coeff_reg_trans_ratio=2, use_reg_trans=False, init_coeff_reg_trans=1, use_spectral_norm=False, eps_sdf=0.01, use_sdf=False,
                 hidden_features=256, nb_hidden_layers=3, relu=False, hartley=False, loss_type='l2', vae_param=0, freeze_encoder=None, impose_symmetry=False,
                 start_ep_learn_het=0, start_ep_learn_het_c2=400, nb_channels=1, lr_pose_net=10**-4, lr_siren=5*10**-6, nb_epochs_each_phases_ACE_Het = None,
                 **rot_args):
        self.params= locals()

        self.nb_epochs_each_phases_ACE_Het = nb_epochs_each_phases_ACE_Het  # when not None, it is a tuple containing 2 elements.
                                    # The first one is the number of epochs in each cycle of training Image to Image,
                                     # the second one is the number of epochs in each cycle of training Pose to Pose

"""
,
            nb_dim_het = 1,
            alphas = np.arange(0.1, 1, 0.1),
            impose_symmetry = False,
            known_rot = True,
            known_trans = True,
            #fd = f'no_heterogeneity_s_{size}'
            anis = 3,
            alpha = 1,
            nb_epochs = 2000,
            start_ep_learn_het = 0,
            x = 200,
            nb_channels = 1,
            #fd = "2_channels_different_max_length_stop_channel_2_z"
            #fd = "views_more_visible_symmetry"
            #pth_views = f'{PATH_PROJECT_FOLDER}/results_deep_learning/heterogene_views/{fd}'
            nb_epochs_each_phases_ACE_Het = None,
            #init = f'{pth_views}/gt_dilated/view_600.0_.tif'
            init = 'random',
            #init = f'{PATH_PROJECT_FOLDER}/article/illustrations/illustr_views/real_data/ours/ab_init/recons_registered_2.tif'
            nb_views = 250,
            impose_cylinder = False,
            particule = 'recepteurs_AMPA'}
"""
