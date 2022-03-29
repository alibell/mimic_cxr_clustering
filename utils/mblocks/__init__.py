# Functions needed to create M-Blocks classifiers

from torch import nn
import torch

class M_Block (nn.Module):
    """
        This block merge together two tensor of 2 images + layers, the left one which has higher dimension then the right one
        Left : (n, n_layers_left, height_left, width_left)
        Right : (n, n_layers_right, height_right, width_right)

        height_left >= height_right
        width_left >= width_right

        Output:
        -------
        A tensor : (n, n_layers_left + n_layers_right, height_right, width_right)
    """

    def __init__ (self, n_layers_left, n_layers_right) :
        super().__init__()

        self.network_left = nn.Sequential(
            nn.Conv2d(n_layers_left, n_layers_left, kernel_size=(3,3), padding="same", bias=False),
            nn.ReLU()
        )

        self.network_right = nn.Sequential(
            nn.Conv2d(n_layers_right, n_layers_right, kernel_size=(3,3), padding="same", bias=False),
            nn.ReLU()
        )

        merged_size = int((n_layers_right+n_layers_left)/2)
        self.merge_networks = nn.Sequential(
            nn.Conv2d(n_layers_right+n_layers_left, merged_size, kernel_size=(3,3), padding="same", bias=False),
            nn.BatchNorm2d(merged_size),
            nn.ReLU()
        )

    def forward(self, left, right):

        left_processed = self.network_left(left)
        right_processed = self.network_right(right)

        meanpool = nn.AdaptiveAvgPool2d(right_processed.shape[-2:])
        left_processed_maxpool = meanpool(left_processed)

        output = torch.cat([
            left_processed_maxpool,
            right_processed
        ], axis=1)

        output = self.merge_networks(output)

        return output
    