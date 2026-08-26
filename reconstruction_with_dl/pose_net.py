import numpy as np
import torch

from torch import cos, sin
from reconstruction_with_dl.encoders.hollyNd import Net3d
import random
"""
from pycls.models.regnet import RegNet
from pycls.models.effnet import EffNet
from pycls.models.resnet import ResNet
from pycls.models.vit import ViT
"""
from reconstruction_with_dl.encoders.VGG16 import CNNEncoderVGG16
from reconstruction_with_dl.encoders.layers import *
from pytorch3d.transforms import axis_angle_to_matrix, euler_angles_to_matrix, rotation_6d_to_matrix, quaternion_to_matrix

rot_rep_dict = {'axis_angle':(3, axis_angle_to_matrix), 'euler':(3,euler_angles_to_matrix), 'quaternion':(4,quaternion_to_matrix), '6d':(6, rotation_6d_to_matrix)}
#encoders_dict = {'holly':Net3d, 'RegNet':RegNet, 'EffNet':EffNet, 'vgg':CNNEncoderVGG16, 'ResNet':ResNet, 'ViT':ViT}
encoders_dict = {'holly':Net3d, 'vgg':CNNEncoderVGG16}
"""
 im_size = 45**3
max bs : 
holly : 8
RegNet : 
EffNet :
VGG :
"""
#TO DO find bug in ResNet and ViT, include cryoAI encoder


def to_numpy(x):
    return x.cpu().detach().numpy() if x is not None else None


class PosesPredictor(nn.Module):
    """generates the encoder architecture, a network that takes as input a batch
    of views and returns the associated poses and confomrational parameters"""
    def __init__(self,im_size, params, nb_dim) -> None:
        super().__init__()
        self.rot_representation = params["rot_representation"]
        self.rot_rep_args = params["rot_args"]
        nb_features_rot = rot_rep_dict[self.rot_representation][0]
        self.nb_dim = nb_dim
        self.heterogeneite = params["heterogeneity"]

        # extracteur de features
        if params["encoder_name"] == 'vgg':
            self.encoder = CNNEncoderVGG16(nb_dim=nb_dim)
        elif params["encoder_name"] == 'holly':
            self.encoder = Net3d(params, nb_dim)
        else:
            pass
        # self.encoder = encoders_dict[params.encoder_name](params, nb_dim=nb_dim)
        # in_features = self.encoder.size_features
        cc = torch.rand(tuple([params["nb_channels"]]+[im_size]*nb_dim)).unsqueeze(0)
        in_features = self.encoder(cc).shape[1]

        use_spect_norm = params["use_sepctral_norm"]
        # couches densément connecté prédisant les rotations
        self.fc1 = linear(in_features, 256, use_spect_norm)
        self.fc2 = linear(256, nb_features_rot, use_spect_norm)

        #couches densément connectée présisant les translations
        self.fc1_trans = linear(in_features,256, use_spect_norm)
        self.fc2_trans = linear(256, nb_dim, use_spect_norm)

        # couches densément connectées préisant le param d'hétérogéité
        self.fc1_heterogeneite_param = nn.Linear(in_features, 256)
        self.fc2_heterogeneite_param = nn.Linear(256, params["nb_dim_het"])

        # couches densément connectées préisant le param d'hétérogéité du deuxième canal s'il y en a un
        self.fc1_heterogeneite_param_c2 = nn.Linear(in_features, 256)
        self.fc2_heterogeneite_param_c2 = nn.Linear(256, params["nb_dim_het"])

        self.layers_fc = [self.fc1, self.fc2, self.fc1_heterogeneite_param, self.fc2_heterogeneite_param,
                          self.fc1_heterogeneite_param_c2, self.fc2_heterogeneite_param_c2]
        self.layers_fc2 = [self.fc1_trans, self.fc2_trans]
        for l in self.layers_fc:
            torch.nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(random.random() * 0.001)
        for l in self.layers_fc2:
            torch.nn.init.xavier_uniform(l.weight, gain=params["init_gain"])
            #l.weight.data.fill_((random.random()-0.5)*params.init_gain * np.sqrt(6/256+nb_dim))
            l.bias.data.fill_(random.random() * 0.001)

        #self.seq = nn.Sequential(*self.encoder.seq, self.fc1, nn.LeakyReLU(), self.fc2)
        if params["batch_norm_rot"]:
            self.rot_regressor = nn.Sequential(self.fc1, nn.LeakyReLU(), self.fc2, nn.BatchNorm1d(nb_features_rot))
        else:
            self.rot_regressor = nn.Sequential(self.fc1, nn.LeakyReLU(), self.fc2)
        if not params["batch_norm_trans"]:
            self.trans_regressor = nn.Sequential(self.fc1_trans, nn.LeakyReLU(), self.fc2_trans) #nn.BatchNorm1d(3), ModifTanh(0.2))
        else:
            self.trans_regressor = nn.Sequential(self.fc1_trans, nn.LeakyReLU(), self.fc2_trans, nn.BatchNorm1d(nb_dim))#, ModifTanh(0.2))
        self.heterogeneite_regressor = nn.Sequential(self.fc1_heterogeneite_param, nn.LeakyReLU(), self.fc2_heterogeneite_param)
        self.heterogeneite_regressor_c2 = nn.Sequential(self.fc1_heterogeneite_param_c2, nn.LeakyReLU(), self.fc2_heterogeneite_param_c2)
        self.params = params
        #self.rot_regressor = MLP([in_features, 256, 3])
        #self.trans_regressor = MLP([in_features, 256, 3])
        
    def reinit_rot_params(self):
        print('re init rot net')
        for l in [self.fc1, self.fc2]:
            torch.nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(random.random() * 0.001)

    def forward(self, x, test=False, pass_het=True, known_rot=False, known_trans=False):
        #rot, trans = self.encoder(x)
        """
        features = self.encoder(x)
        rot = self.rot_regressor(features)
        if not self.known_trans:
            trans = self.trans_regressor(features)
        else:
            trans = None
        """
        if not known_rot or not known_trans or self.heterogeneite:
            features = self.encoder(x) # get the features vector
        if known_rot:
            rot = None
            est_rot_mat = None
        else:
            rot = self.rot_regressor(features) # get the estimated rot vector from features
            if self.nb_dim == 2:
                # print('est rot', torch.remainder(est_rot, 2*np.pi))
                # print('est rot', est_rot)
                est_rot_mat = torch.stack([cos(rot), -sin(rot), sin(rot), cos(rot)]).T.view((-1, 2, 2))
            else:
                est_rot_mat = rot_rep_dict[self.rot_representation][1](rot, **self.rot_rep_args) # transforms the estimated
                # rotation vector into a rotation matrix
        if known_trans:
            trans = None
        else:
            trans = self.params["coeff_trans"]*self.trans_regressor(features) # estimates the translation parameters
            # from the features vector
        if self.heterogeneite and pass_het:
            heterogeneite_val = self.predict_heterogeneity(self.heterogeneite_regressor, features, test) # predicts the heterogeneity
                # params from the features parameters
        else:
            # heterogeneite_val = None
            heterogeneite_val = torch.zeros((x.shape[0],self.params["nb_dim_het"])).cuda(self.params["device"])
            # print('trans', trans)
        #trans = self.trans_regressor(features)
        return est_rot_mat, trans, heterogeneite_val

    def predict_heterogeneity(self, heterogeneity_regressor, features, test):
        # If the symmetric loss is used, each input view is duplicated 4 times. However, we want the conformation parameter associated with these 4 duplicates to be identical.
        # This is why we only input a quarter of the feature vector into the network that predicts heterogeneity, then we duplicate the output 4 times.
        h = 4 if self.nb_dim == 3 else 2
        if self.params["sym_loss"] and not test:
            features_first_batches = features[:features.shape[0] // h]
        else:
            features_first_batches = features
        heterogeneite_val = heterogeneity_regressor(features_first_batches)
        if self.params["sym_loss"] and not test:
            heterogeneite_val = torch.cat([heterogeneite_val] * h)
        return heterogeneite_val

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    pass
    

    

        
        







    

    





