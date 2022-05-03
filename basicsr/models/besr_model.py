import importlib
import torch
from collections import OrderedDict
from copy import deepcopy

from basicsr.models.archs import define_network
from basicsr.models.sr_model import SRModel
from basicsr.models.archs.WPT_arch import DWT
from basicsr.models.archs.Gau_Lap_arch import LaplacianConv, GaussianBlurConv

loss_module = importlib.import_module('basicsr.models.losses')


class BESRModel(SRModel):
    """BESR model for single image super-resolution."""

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('lpips_opt'):
            import basicsr.models.losses.LPIPS.models.dist_model as dm
            self.model_LPIPS = dm.DistModel()
            self.model_LPIPS.initialize(model='net-lin', net='alex', use_gpu=True)
            self.lpips_weight = train_opt['lpips_opt']['loss_weight']
        else:
            self.model_LPIPS = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        if self.model_LPIPS:
            loss_LPIPS, _ = self.model_LPIPS.forward_pair(self.gt * 2 - 1, self.output * 2 - 1)
            loss_LPIPS = torch.mean(loss_LPIPS) * self.lpips_weight
            l_total += loss_LPIPS
            loss_dict['l_lpips'] = loss_LPIPS

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

