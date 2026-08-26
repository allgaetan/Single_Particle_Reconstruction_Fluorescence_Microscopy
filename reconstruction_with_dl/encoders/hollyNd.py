import imp
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from reconstruction_with_dl.encoders.layers import batchnorm_layer, conv_layer



def conv_size(shape, padding=0, kernel_size=5, stride=1) -> int:
    """
    Return the size of the convolution layer given a set of parameters
    Parameters
    ----------
    x : int
        The size of the input tensor
    padding: int
        The conv layer padding - default 0
    
    kernel_size: int
        The conv layer kernel size - default 5
    stride: int 
        The conv stride - default 1
    """
    return tuple([int((shape[i] - kernel_size + 2 * padding) / stride + 1) for i in range(len(shape))])


def num_flat_features(x):
    """
    Return the number of features of this neural net layer,
    if it were flattened.
    Parameters
    ----------
    x : torch.Tensor
        The layer in question.
    Returns
    -------
    int
        The number of features
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class Flatten(nn.Module):
    """
    An nn module that flattens input so it can be passed to the 
    fully connected layers.
    """
    def forward(self, x):
        return x.view(x.size()[0], -1)
        #return x.view(-1, num_flat_features(x))


class Net3d(nn.Module):
    """The defininition of our convolutional net that reads our images
    and spits out some angles and attempts to figure out the loss
    between the output and the original simulated image.
    """

    def __init__(self, params, nb_dim=3):
        super(Net3d, self).__init__()
        # Conv layers
        self.batch1 = batchnorm_layer(16, nb_dim)
        self.batch2 = batchnorm_layer(32, nb_dim)
        self.batch2b = batchnorm_layer(32, nb_dim)
        self.batch3 = batchnorm_layer(64, nb_dim)
        self.batch3b = batchnorm_layer(64, nb_dim)
        self.batch4 = batchnorm_layer(128, nb_dim)
        self.batch4b = batchnorm_layer(128, nb_dim)
        self.batch5 = batchnorm_layer(256, nb_dim)
        self.batch5b = batchnorm_layer(256, nb_dim)
        self.batch6 = batchnorm_layer(256, nb_dim)

        # Added more conf layers as we aren't using maxpooling
        # TODO - we only have one pseudo-maxpool at the end
        # TODO - do we fancy some drop-out afterall?
        use_sepctral_norm = params["use_sepctral_norm"]
        self.conv1 = conv_layer(nb_dim, use_sepctral_norm, params["nb_channels"], 16, 5, stride=2, padding=2)
        #csize = conv_size(im_shape, padding=2, stride=2)

        self.conv2 = conv_layer(nb_dim, use_sepctral_norm,16, 32, 3, stride=1, padding=1)
        #csize = conv_size(csize, padding=1, stride=1, kernel_size=3)

        self.conv2b = conv_layer(nb_dim,use_sepctral_norm,32, 32, 2, stride=2, padding=1)
        #csize = conv_size(csize, padding=1, stride=2, kernel_size=2)

        self.conv3 = conv_layer(nb_dim,use_sepctral_norm,32, 64, 3, stride=1, padding=1)
        #csize = conv_size(csize, padding=1, stride=1, kernel_size=3)

        self.conv3b = conv_layer(nb_dim,use_sepctral_norm,64, 64, 2, stride=2, padding=1)
        #csize = conv_size(csize, padding=1, stride=2, kernel_size=2)

        self.conv4 = conv_layer(nb_dim,use_sepctral_norm,64, 128, 3, stride=1, padding=1)
        #csize = conv_size(csize, padding=1, stride=1, kernel_size=3)

        self.conv4b = conv_layer(nb_dim,use_sepctral_norm,128, 128, 2, stride=2, padding=1)
        #csize = conv_size(csize, padding=1, stride=2, kernel_size=2)

        self.conv5 = conv_layer(nb_dim,use_sepctral_norm,128, 256, 3, stride=1, padding=1)
        #csize = conv_size(csize, padding=1, stride=1, kernel_size=3)

        self.conv5b = conv_layer(nb_dim,use_sepctral_norm,256, 256, 2, stride=2, padding=1)
        #csize = conv_size(csize, padding=1, stride=2, kernel_size=2)

        self.conv6 = conv_layer(nb_dim,use_sepctral_norm,256, 256, 3, stride=1, padding=1)
        #csize = conv_size(csize, padding=1, stride=1, kernel_size=3)
        
        # Fully connected layers
        last_filter_size = 256
        """
        in_fc1 = 1
        for i in range(len(csize)):
            in_fc1 *= csize[i]
        in_fc1 *= last_filter_size
        self.size_features = in_fc1
        """
        #self.fc1 = nn.Linear(in_fc1, 256)
        #self.fc2 = nn.Linear(256, num_params)
        
        self.seq = nn.Sequential(
            self.conv1,
            self.batch1,
            nn.LeakyReLU(),
            self.conv2,
            self.batch2,
            nn.LeakyReLU(),
            self.conv2b,
            self.batch2b,
            nn.LeakyReLU(),
            self.conv3,
            self.batch3,
            nn.LeakyReLU(),
            self.conv3b,
            self.batch3b,
            nn.LeakyReLU(),
            self.conv4,
            self.batch4,
            nn.LeakyReLU(),
            self.conv4b,
            self.batch4b,
            nn.LeakyReLU(),
            self.conv5,
            self.batch5,
            nn.LeakyReLU(),
            self.conv5b,
            self.batch5b,
            nn.LeakyReLU(),
            self.conv6,
            self.batch6,
            nn.LeakyReLU(),
            Flatten() #,
            #self.fc1,
            #nn.LeakyReLU(),
            #self.fc2
        )
        #self.seq = nn.Sequential(*self.seq, self.fc1, nn.LeakyReLU(), self.fc2)

        self.layers = [
            self.conv1,
            self.conv2,
            self.conv2b,
            self.conv3,
            self.conv3b,
            self.conv4,
            self.conv4b,
            self.conv5,
            self.conv5b,
            self.conv6,
            #self.fc1,
            #self.fc2,
        ]

        # Specific weight and bias initialisation
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(random.random() * 0.001)

    def __iter__(self):
        return iter(self.layers)

    def __next__(self):
        """
        Return the 'next' layer in the network
        """
        if self._lidx > len(self.layers):
            self._lidx = 0
            raise StopIteration

        rval = self.layers[self._lidx]
        self._lidx += 1
        return rval

    def get_render_params(self):
        """
        Return the resulting renderer parameters.
        """
        return self._final

    def forward(self, source: torch.Tensor):
        """
        Our forward pass. We take the input image (x), the
        vector of points (x,y,z,w) and run it through the model. Offsets
        is an optional list the same size as points.
        Initialise the model.
        Parameters
        ----------
        source : torch.Tensor
            The source image, as a tensor.
        points : PointsTen
            The points we are predicting.
        Returns
        -------
        None
        """
        
        out = self.seq(source)
        """
        rot = out[:, :3]
        if not self.known_trans:
            trans = out[:, 3:]
        else:
            trans = None
        
        """
        return out

if __name__ == '__main__':
    from classes_with_parameters import ParametersLearningSetup
    params = ParametersLearningSetup()
    holly = Net3d(params)
    torch.save(holly, 'neural_net.pt')

    #print('sum', summary(holly, (1,50,50,50)))

