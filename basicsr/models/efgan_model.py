import importlib
import torch
import numpy as np
from collections import OrderedDict

from basicsr.models.archs import define_network
from basicsr.models.srgan_model import SRGANModel
from basicsr.models.archs.WPT_arch import DWT
from basicsr.models.archs.Gau_Lap_arch import LaplacianConv, GaussianBlurConv

loss_module = importlib.import_module('basicsr.models.losses')


class EFGANModel(SRGANModel):
    """EFGAN model for single image super-resolution."""

    def init_training_settings(self):
        train_opt = self.opt['train']

        # define network wavelet_dec
        self.wavelet_dec = DWT()

        # define network Laplacian and GaussianBlur
        self.Laplacian = LaplacianConv()
        self.GaussianBlur = GaussianBlurConv

        # define network net_d
        self.net_d = networks.define_net_d(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path,
                              self.opt['path']['strict_load'])

        self.net_g.train()
        self.net_d.train()

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

        if train_opt.get('gan_opt'):
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)
            self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(self.device)

        if train_opt.get('wavelet_opt'):
            wavelet_type = train_opt['wavelet_opt'].pop('type')
            cri_wavelet_cls = getattr(loss_module, wavelet_type)
            self.cri_wave = cri_wavelet_cls(**train_opt['wavelet_opt']).to(
                self.device)
        else:
            self.cri_wave = None

        if train_opt.get('texture_opt'):
            texture_type = train_opt['texture_opt'].pop('type')
            cri_texture_cls = getattr(loss_module, texture_type)

            self.cri_text = cri_texture_cls(**train_opt['texture_opt']).to(
                self.device)
        else:
            self.cri_text = None

        if train_opt.get('gau_lap_opt'):
            gau_lap_type = train_opt['gau_lap_opt'].pop('type')
            cri_gau_lap_cls = getattr(loss_module, gau_lap_type)

            self.cri_gau_lap = cri_gau_lap_cls(**train_opt['gau_lap_opt']).to(
                self.device)
        else:
            self.cri_gau_lap = None

        self.net_d_iters = train_opt['net_d_iters'] if train_opt[
            'net_d_iters'] else 1
        self.net_d_init_iters = train_opt['net_d_init_iters'] if train_opt[
            'net_d_init_iters'] else 0

        # control how much emphasis is put on L(G(z_D)) during gradient descent.
        self.k = 0
        self.M_global = AverageMeter()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        recon_real = self.net_d(self.gt)
        self.output = self.net_g(self.lq)
        recon_fake = self.net_d(self.output.detach())

        wavelet_recon_real = self.wavelet_dec(recon_real)
        wavelet_recon_fake = self.wavelet_dec(recon_fake)
        wavelet_fake = self.wavelet_dec(self.output.detach())
        wavelet_gt = self.wavelet_dec(self.gt)

        Gau_Lap_recon_real = self.GaussianBlur(self.Laplacian(recon_real))
        Gau_Lap_recon_fake = self.GaussianBlur(self.Laplacian(recon_fake))
        Gau_Lap_real = self.GaussianBlur(self.Laplacian(self.gt))
        Gau_Lap_fake = self.GaussianBlur(self.Laplacian(self.output.detach()))

        l_d_mse_real = self.cri_pix(recon_real, self.gt)
        l_d_mse_fake = self.cri_pix(recon_fake, self.output)
        l_d_wavelet_real = self.cri_wave(wavelet_gt, wavelet_recon_real)
        l_d_wavelet_fake = self.cri_wave(wavelet_fake, wavelet_recon_fake)
        l_d_texture_real = self.cri_text(wavelet_gt, wavelet_recon_real)
        l_d_texture_fake = self.cri_text(wavelet_fake, wavelet_recon_fake)
        l_d_Gau_Lap_real = self.cri_gau_lap(Gau_Lap_recon_real, Gau_Lap_real)
        l_d_Gau_Lap_fake = self.cri_gau_lap(Gau_Lap_recon_fake, Gau_Lap_fake)

        l_d_real = l_d_mse_real + l_d_texture_real + l_d_wavelet_real + l_d_Gau_Lap_real
        l_d_fake = l_d_mse_fake + l_d_texture_fake + l_d_wavelet_fake + l_d_Gau_Lap_fake

        l_d_total = l_d_real - self.k * l_d_fake
        l_d_total.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        #loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        #loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        #self.output = self.net_g(self.lq)

        l_g_total = 0
        if (current_iter % self.net_d_iters == 0
                and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(
                    self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # gan loss
            recon_fake = self.net_d(self.output)

            wavelet_recon_fake = self.wavelet_dec(recon_fake)
            wavelet_fake = self.wavelet_dec(self.output)

            Gau_Lap_recon_fake = self.GaussianBlur(self.Laplacian(recon_fake))
            Gau_Lap_fake = self.GaussianBlur(self.Laplacian(self.output))

            l_g_mse_fake = self.cri_pix(recon_fake, self.output)
            l_g_wavelet_fake = self.cri_wave(wavelet_fake, wavelet_recon_fake)
            l_g_texture_fake = self.cri_text(wavelet_fake, wavelet_recon_fake)
            l_g_Gau_Lap_fake = self.cri_gau_lap(Gau_Lap_recon_fake, Gau_Lap_fake)

            l_g_gan = l_g_mse_fake + l_g_texture_fake + l_g_wavelet_fake + l_g_Gau_Lap_fake

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        balance = (self.opt["gamma"] * l_d_real - l_d_fake).item()
        self.k = min(max(self.k + self.opt['lambda_k'] * balance, 0), 1)
        measure = l_d_real.item() + np.abs(balance)
        self.M_global.update(measure, self.gt.size(0))
        loss_dict['balance'] = balance
        loss_dict['k'] = self.k
        loss_dict['M_global'] = self.M_global.avg


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count