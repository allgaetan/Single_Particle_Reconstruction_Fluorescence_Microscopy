
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from data_generation.generate_data import generate_data
from reconstruction_with_dl.test_params.default_params import params_data_gen
from reconstruction_with_dl.data_set_views import ViewsRandomlyOrientedSimData
from reconstruction_with_dl.pose_net import to_numpy


torch.set_default_dtype(torch.float32)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, relu=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.relu = relu

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        if self.relu:
            return torch.relu(self.linear(input))
        else:
            return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions

        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    def reset_parameters(self):
        self.linear.reset_parameters()


class Siren(nn.Module):
    """Représentation implicite sous forme de réseaux de neurones
    (voir Implicit Neural Representations with Periodic Activation Functions, Vincent Sitzmann Neurips 2020).
    Il s'agit d'un réseau densément connecté avec des fonctions d'activation sinusoïdales. Il prend en entrée les 3 coordonnées d'un voxels,
    concaténés au paramètres de conformation et renvoie l'intensité associée."""
    def __init__(self, in_features, hidden_features=256, hidden_layers=3, out_features=1, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., relu=False):
        """in_features : nombre neurones de la couche d'entrée (égal à 3 + la dimension de l'espace latent)
        hidden_features : nombre de neurones dans les couches intermédiaires
        hidden_layer : nombre de couches cachées
        out_features : nombre de neurones en sortie (égal au nombre de canaux)
        first_omega_0 : pulsation des fonctions d'activations sinusoïdales de la première couche
        hidden_omega_0 : pulsation des fonctions d'activations sinusoïdales des couches suivantes"""
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0, relu=relu))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0, relu=relu))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0, relu=relu))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords
    
    def reset_parameters(self):
        for l in self.net:
            l.reset_parameters()

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations





def fit_SIREN_with_image(im, params_learning_setup, bs=5, lr=1e-5):
    """
    fonction : fitter une représentation implicite SIREN sur l'image 3D im. Si on se contente de fitter les coordonnées d'une
    grille 3D unique avec les valeures associées de l'image im, les valeures associées à des pixels de coordonnées intermésiaires
    ne seront pas correctement apprises. C'est pourquoi, cette fonction génère 100 rotations de l'images originales et fitte
    les 100 images rotationnées avec les 100 grilles rotationnées des mêmes angles.
    """
    device = params_learning_setup["device"]
    nb_dim = len(im.shape)
    size = im.shape[0]
    params_data_gen['nb_views'], params_data_gen["size"] = 50, size
    print('size', size)
    params_data_gen["no_psf"] = True
    views, rot_vecs, transvecs, rot_mats, _, _, _ = generate_data(im, params_data_gen)
    im_data_set = ViewsRandomlyOrientedSimData(np.array(views), rot_mats, rot_vecs, transvecs, [0]*len(views), size, nb_dim,  ['' for _ in range(len(views))])
    dataloader = DataLoader(im_data_set, batch_size=bs, pin_memory=True, num_workers=0)
    nb_dim_siren = nb_dim if not params_learning_setup["heterogeneity"] else nb_dim + 1
    siren = Siren(nb_dim_siren, first_omega_0=params_learning_setup["omega"],
                  hidden_omega_0=params_learning_setup["omega"], hidden_features=params_learning_setup["nb_hidden_features"],
                  hidden_layers=params_learning_setup["nb_hidden_layers"], relu=params_learning_setup["relu"])
    siren = siren.cuda(device)
    optim = torch.optim.Adam(lr=lr, params=siren.parameters())
    print('start fitting')
    #print('het', heterogenity)
    losses = []
    for _ in range(100):
        loss_iter = 0
        for v,d in enumerate(dataloader):
            grid_view, view, _, _, _, _, _, _ = d
            grid_view = grid_view.cuda(device)
            view = view.cuda(device)
            if params_learning_setup["heterogeneity"]:
                het_val = torch.zeros(bs, 1).cuda(params_learning_setup["device"])
                est_heterogeneite_repeated = het_val.unsqueeze(2).repeat(1, grid_view.shape[1], 1)
                concat_grid_heterogeneity = torch.cat((grid_view, est_heterogeneite_repeated), 2)
            else:
                concat_grid_heterogeneity = grid_view
            model_output, _ = siren(concat_grid_heterogeneity)
            #print('model output shape', model_output.shape)
            if params_learning_setup["loss_type"] == 'l2':
                loss = ((model_output.squeeze() - view.view(-1,1, im.shape[0]**len(im.shape)).squeeze())**2).mean()
            else:
                loss = abs(model_output.squeeze() - view.view(-1, 1, im.shape[0] ** len(im.shape)).squeeze()).mean()
            loss_iter += loss
            optim.zero_grad()
            loss.backward()
            optim.step()
        print('loss', loss_iter)
        losses.append(to_numpy(loss_iter))
    fitted_img = model_output[0].view(*([size] * nb_dim))
    #if params_learning_setup.hartley:
        #fitted_img = idht(fitted_img)
    return siren, to_numpy(fitted_img), losses





