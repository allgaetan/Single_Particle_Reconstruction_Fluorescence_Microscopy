import numpy as np

class ParametersMainAlg:
    """Hyperparameter of the function gd_importance_sampling_3d, in file gradient_descent_importance_sampling.py, folder
    learning_algorithms. """
    def __init__(self, M_axes=360**2, M_rot=360, dec_prop=1.2, init_unif_prop=1, coeff_kernel_axes=50., coeff_kernel_rot=5., eps=0, lr=0.1,
                 N_axes=25, N_rot=20, prop_min=0, interp_order=3, N_iter_max=20, gaussian_kernel=True, N_iter_with_unif_distr=None,
                 epochs_of_suppression=None, proportion_of_views_suppressed=None, convention='ZXZ'):
        self.params = locals()
        self.M_axes = M_axes  #number of axes in the discretization of the 3d rotations
        self.M_rot = M_rot # number of third angle in the discretization of the 3d rotations
        self.dec_prop = dec_prop # the proporion of uniform distribution is divided by this factor at each iteration
        self.coeff_kernel_axes = coeff_kernel_axes # the coeffictient of the kernel associated to the axes
        self.coeff_kernel_rot = coeff_kernel_rot #the coefficient of the kernel associated to the third angle
        self.eps = eps # theshold that control early stopping
        self.lr = lr # learning rate
        self.N_axes = N_axes # Number of axes randomly drawn at each iteration of the importance sampling
        self.N_rot = N_rot # Number of third angle randomly drawn at each iteration of the importance sampling
        self.prop_min = prop_min # minimum proportion of the uniform part of the importance distribution
        self.interp_order = interp_order # interpolation order used in the 3d rotation
        self.gaussian_kernel = gaussian_kernel # use of gaussian kernel to interpolate the importance distribution
        self.N_iter_with_unif_distr = N_iter_with_unif_distr
        self.epochs_of_suppression = epochs_of_suppression
        self.proportions_of_views_suppressed = proportion_of_views_suppressed
        self.N_iter_max = N_iter_max #maximum number of iteration
        self.init_unif_prop = init_unif_prop # initial value of the uniform distribution
        self.convention = convention #convention used to represent the angles


class ParametersDataGeneration:
    """hyperparameters used to generate the simulated data (function generate_data in folder generate_data.py)"""
    def __init__(self, nb_dim=3, nb_views=10, sig_z=5,sig_xy=1,snr=100,order=3,
                              sigma_trans_ker=0, size=50, max_dilatation_val=None, dilatation_function=None, psf=None, projection=False,
                 rot_vecs=None, no_psf=False, convention='ZXZ', partial_labelling=False, rotation_max=None,
                 **partial_labelling_args):
        self.params = locals()
        self.no_psf = no_psf #boolean, if true use no psf
        self.nb_views = nb_views #number of generated views
        self.sig_z = sig_z #standard deviation of the psf along z axis
        self.sig_xy = sig_xy # standard deviation of the psf in xy plane
        self.snr = snr # signal to noise ration
        self.sigma_trans_ker = sigma_trans_ker # standard deviation of the distribution that generates the translation parameters
        self.size = size # size of one side of a 3d image
        self.grid_step = 2/(size-1)
        self.psf = psf #point spread function. If it is none, it generates a point spread function from the standard deviation provided
        self.partial_labelling = partial_labelling # partial labelling : if True, it simulates partial labelling by removing some small
        # gaussian spots at random location in the image
        self.partial_labelling_args = partial_labelling_args # argument of the partial labelling, if used
        self.projection = projection # if true, 2 projectiosn are generatd instaed of 3d viwes with anisotropic resolutiin
        self.rot_vecs = rot_vecs
        self.convention = convention
        self.nb_dim = nb_dim # number of dimension of the simulated data, usually equals 3
        self.order= order # order of interpolation
        self.max_dilatation_val = max_dilatation_val
        self.dilatation_function = dilatation_function
        self.rotation_max = rotation_max

    def get_cov_psf(self):
        cov_PSF = self.grid_step ** 2 * np.eye(self.nb_dim)
        cov_PSF[0, 0] *= self.sig_z ** 2
        for i in range(1, self.nb_dim):
            cov_PSF[i, i] *= self.sig_xy ** 2
        return cov_PSF

    def get_psf(self):
        if self.psf is None:
            cov_psf = self.get_cov_psf()
            grid = make_grid(self.size, self.nb_dim)
            psf = nd_gaussian(grid, np.zeros(self.nb_dim), cov_psf, self.nb_dim)
            psf /= np.sum(psf)
            return psf
        else:
            return self.psf/np.sum(self.psf)


class ParametersGMM:
    """hyperparameters of the gaussian representation of the volume"""
    def __init__(self, nb_gaussians_init = 10, nb_gaussians_ratio = 4, sigma_init = 0.2, sigma_ratio = 1.4,
            nb_steps = 4, threshold_gaussians = 0.01, unif_prop_mins = [0.5, 0.25, 0.125, 0], init_with_views=True):
        self.nb_gaussians_init = nb_gaussians_init
        self.nb_gaussians_ratio = nb_gaussians_ratio
        self.sigma_init = sigma_init
        self.sigma_ratio = sigma_ratio
        self.nb_steps = nb_steps
        self.threshold_gaussians = threshold_gaussians
        self.unif_prop_mins = unif_prop_mins
        self.init_with_views = init_with_views

class ParametersLearningSetup:
    def __init__(self, heterogeneity=False, nb_dim_het=1, encoder_name='holly', rot_representation='6d', use_sym_loss=False,
                device=0, nb_epochs=100, bs=8, init='random', omega=30, coeff_trans=0.05, init_gain=1, batch_norm_rot=False, batch_norm_trans=False,
                coeff_reg_trans_ratio=2, use_reg_trans=False, init_coeff_reg_trans=1, use_spectral_norm=False, eps_sdf=0.01, use_sdf=False,
                 hidden_features=256, nb_hidden_layers=3, relu=False, hartley=False, loss_type='l2', vae_param=0, freeze_encoder=None, impose_symmetry=False,
                 start_ep_learn_het=0, start_ep_learn_het_c2=400, nb_channels=1, lr_pose_net=10**-4, lr_siren=5*10**-6, nb_epochs_each_phases_ACE_Het = None,
                 **rot_args):
        self.params= locals()
        self.start_ep_learn_het = start_ep_learn_het
        self.start_ep_learn_het_c2 = start_ep_learn_het_c2
        self.heterogeneity = heterogeneity
        self.nb_dim_het = nb_dim_het
        self.encoder_name = encoder_name
        self.rot_representation = rot_representation
        self.rot_args = rot_args
        self.use_sym_loss = use_sym_loss
        self.nb_epochs = nb_epochs
        self.nb_channels = nb_channels
        self.bs = bs
        self.device = device
        self.init = init
        self.batch_norm_rot = batch_norm_rot
        self.coeff_trans = coeff_trans
        self.batch_norm_trans = batch_norm_trans
        self.coeff_reg_trans_ratio = coeff_reg_trans_ratio
        self.init_gain = init_gain
        self.use_reg_trans= use_reg_trans
        self.init_coeff_reg_trans = init_coeff_reg_trans
        self.use_spectral_norm = use_spectral_norm
        self.eps_sdf = eps_sdf
        self.use_sdf = use_sdf
        self.omega = omega
        self.nb_hidden_layers = nb_hidden_layers
        self.hidden_features = hidden_features
        self.relu = relu
        self.loss_type = loss_type
        self.hartley = hartley
        self.vae_param = vae_param
        self.freeze_encoder = freeze_encoder
        self.impose_symmetry = impose_symmetry
        self.lr_pose_net = lr_pose_net
        self.lr_siren = lr_siren
        self.nb_epochs_each_phases_ACE_Het = nb_epochs_each_phases_ACE_Het  # when not None, it is a tuple containing 2 elements.
                                    # The first one is the number of epochs in each cycle of training Image to Image,
                                     # the second one is the number of epochs in each cycle of training Pose to Pose




