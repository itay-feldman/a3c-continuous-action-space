import torch
import torch.nn as nn
import numpy as np


class ModelA3C(nn.Module):
    def __init__(self, input_shape, output_size):
        super(ModelA3C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 32, kernel_size=4, padding=1),
            nn.PReLU()
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.PReLU()
        )

        base_shape = self._get_base_shape(input_shape)

        print('Convolution Base Shape', base_shape)

        self.mu = nn.Sequential(
            nn.Linear(base_shape, output_size),
            nn.Tanh()
        )

        self.var = nn.Sequential(
            nn.Linear(base_shape, output_size),
            nn.Softplus()
        )

        self.value = nn.Linear(base_shape, 1)

    def forward(self, data):
        # split data to 2 parts, rgb images and semantice images
        # data = (data[:,0,:,:,:] + data[:,1,:,:,:]) / 2  # we average the data, this is done for a few reasons, among them are memory limits
        # print('data shape', data.shape, rgb.shape, semantic.shape)
        # print('added shape', data.shape)
        data = self.conv(data).flatten(start_dim=1)  # convert (1, 3D) to (1, x)
        # print('conv base shape', base_out.shape)
        return self.mu(data), self.var(data), self.value(data)

    def _get_base_shape(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
