import torch
import torch.nn as nn
import numpy as np

class LaplacianConv(nn.Module):
    """
    laplace算子作为边缘检测，是各方向的二阶导数

    其卷积模板为:
    0   1   0
    1   -4  1
    0   1   0

    或拓展模板
    1   1   1
    1   -8  1
    1   1   1
    """
    def __init__(self):
        super(LaplacianConv, self).__init__()
        laplace_kernel = np.array([[1, 1, 1],
                                   [1, -8, 1],
                                   [1, 1, 1]])
        conv_kernel = torch.FloatTensor(np.stack((laplace_kernel, laplace_kernel, laplace_kernel), 0)).unsqueeze(0)
        self.weight = nn.Parameter(data=conv_kernel, requires_grad=False)

    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, stride=1, padding=1)
        return x

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        # print("channels: ", channels.shape)
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)    # (H, W) -> (1, 1, H, W)
        #kernel = kernel.expand((int(channels), 1, 5, 5))
        kernel = kernel.repeat(3, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x