import torch
import numpy as np

import basicsr.models.losses.LPIPS.models.dist_model as dm

def calculate_lpips(img1,
                   img2):
    model_LPIPS = dm.DistModel()
    model_LPIPS.initialize(model='net-lin', net='alex', use_gpu=True)

    dist = model_LPIPS.forward(im2tensor(img1), im2tensor(img2))
    return float(dist[0])

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
