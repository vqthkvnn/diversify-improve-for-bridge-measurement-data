# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch.nn as nn
import matplotlib.pyplot as plt

var_size = {
    'emg': {
        'in_size': 5,
        'ker_size': 9,
        'fc_size': 32*44
    }
}


class ActNetwork(nn.Module):
    def __init__(self, taskname):
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=var_size[taskname]['in_size'], out_channels=16, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(
                1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.in_features = var_size[taskname]['fc_size']

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = x.view(-1, self.in_features)
        # plt.figure()
        # plt.plot(x.cpu().detach().numpy()[:, 0])
        # plt.show()
        return x
