import torch
from torch import nn


class L2LossPreFlip(nn.Module):
    """implements the symmetry loss"""
    def __init__(self, nb_dim, loss_type):
        super(L2LossPreFlip, self).__init__()
        self.nb_dim = nb_dim
        assert loss_type == 'l2' or loss_type == 'l1'
        self.loss_type = loss_type

    def forward(self, model_output, gt):
        # model output : 4B, Nc, S, S, S
        # GT : B, Nc, S, S, S
        # gt_permuted = gt.permute(1, 0, 2, 3, 4)  # 1, B, S, S, S

        model_output_reshaped = model_output.reshape(4, model_output.shape[0]//4, model_output.shape[1],
                                                     model_output.shape[2], model_output.shape[3], model_output.shape[4]) # 4, B, Nc, S, S, S

        # Unflip
        model_output_reshaped[1] = torch.flip(model_output_reshaped[1], [3,4])
        model_output_reshaped[2] = torch.flip(model_output_reshaped[2], [2,3])
        model_output_reshaped[3] = torch.flip(model_output_reshaped[3], [2,4])
        if self.loss_type == 'l2':
            distance_double = torch.mean((gt - model_output_reshaped)**2,  (3, 4, 5)) # 4, B, Nc
        else:
            distance_double = torch.mean(torch.abs(gt - model_output_reshaped),  (3, 4, 5))
        min_distances, _ = torch.min(distance_double, 0) # B, Nc
        return min_distances.mean()