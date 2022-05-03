import importlib
import numpy as np
import torch
from collections import OrderedDict
from copy import deepcopy

from basicsr.models.archs import define_network
from basicsr.models.sr_model import SRModel
from basicsr.models.archs.WPT_arch import DWT
from basicsr.models.archs.Gau_Lap_arch import LaplacianConv, GaussianBlurConv

loss_module = importlib.import_module('basicsr.models.losses')

class BEGANModel(SRModel):
    """BESRGAN model for single image super-resolution."""

    def init_training_settings(self):
        train_opt = self.opt['train']

        # weight of waveletLoss, textureLoss, gau_lapLoss
        self.gan_weight = train_opt['gan_opt']['loss_weight']
        self.wavelet_weight_D = train_opt['wavelet_opt']['loss_weight_D']
        self.wavelet_weight_G = train_opt['wavelet_opt']['loss_weight_G']
        self.texture_weight_D = train_opt['texture_opt']['loss_weight_D']
        self.texture_weight_G = train_opt['texture_opt']['loss_weight_G']
        self.gau_lap_weight_D = train_opt['gau_lap_opt']['loss_weight_D']
        self.gau_lap_weight_G = train_opt['gau_lap_opt']['loss_weight_G']

        # define network net_d
        self.net_d = define_network(deepcopy(self.opt['network_d']))
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path,
                              self.opt['path']['strict_load'])

        self.net_g.train()
        self.net_d.train()

        # define losses
        # G pixel loss
        if train_opt['pixel_opt']['loss_weight'] > 0:
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

        # BEGAN Loss
        if train_opt.get('gan_opt'):
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)
            self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(self.device)

        # define network wavelet_dec
        if self.wavelet_weight_D > 0 or self.wavelet_weight_G > 0:
            self.wavelet_dec = DWT()
            self.wavelet_dec = self.model_to_device(self.wavelet_dec)

        # D wavelet loss
        if self.wavelet_weight_D > 0:
            wavelet_type = train_opt['wavelet_opt'].pop('type')
            cri_wavelet_cls = getattr(loss_module, wavelet_type)
            self.cri_wave_D = cri_wavelet_cls(train_opt['wavelet_opt']['loss_weight_D'], train_opt['wavelet_opt']['reduction']).to(
                self.device)
        else:
            self.cri_wave_D = None
        # G wavelet loss
        if self.wavelet_weight_G > 0:
            if wavelet_type == None:
                wavelet_type = train_opt['wavelet_opt'].pop('type')
            cri_wavelet_cls = getattr(loss_module, wavelet_type)
            self.cri_wave_G = cri_wavelet_cls(train_opt['wavelet_opt']['loss_weight_G'], train_opt['wavelet_opt']['reduction']).to(
                self.device)
        else:
            self.cri_wave_G = None

        # D texture loss
        if self.texture_weight_D > 0:
            texture_type = train_opt['texture_opt'].pop('type')
            cri_texture_cls = getattr(loss_module, texture_type)
            self.cri_text_D = cri_texture_cls(train_opt['texture_opt']['loss_weight_D'], train_opt['texture_opt']['reduction']).to(
                self.device)
        else:
            self.cri_text_D = None

        # G texture loss
        if self.texture_weight_G > 0:
            if texture_type == None:
                texture_type = train_opt['texture_opt'].pop('type')
            cri_texture_cls = getattr(loss_module, texture_type)
            self.cri_text_G = cri_texture_cls(train_opt['texture_opt']['loss_weight_G'],
                                              train_opt['texture_opt']['reduction']).to(
                self.device)
        else:
            self.cri_text_G = None

        # define network Laplacian and GaussianBlur
        if self.gau_lap_weight_D > 0 or self.gau_lap_weight_G > 0:
            self.Laplacian = LaplacianConv()
            self.GaussianBlur = GaussianBlurConv()
            self.Laplacian = self.model_to_device(self.Laplacian)
            # self.GaussianBlur = self.model_to_device(self.GaussianBlur)

        # D gau_lap loss
        if self.gau_lap_weight_D > 0:
            gau_lap_type = train_opt['gau_lap_opt'].pop('type')
            cri_gau_lap_cls = getattr(loss_module, gau_lap_type)
            self.cri_gau_lap_D = cri_gau_lap_cls(train_opt['texture_opt']['loss_weight_D'], train_opt['texture_opt']['reduction']).to(
                self.device)
        else:
            self.cri_gau_lap_D = None

        # G gau_lap loss
        if self.gau_lap_weight_G > 0:
            if gau_lap_type == None:
                gau_lap_type = train_opt['gau_lap_opt'].pop('type')
            cri_gau_lap_cls = getattr(loss_module, gau_lap_type)
            self.cri_gau_lap_G = cri_gau_lap_cls(train_opt['texture_opt']['loss_weight_G'],
                                                 train_opt['texture_opt']['reduction']).to(
                self.device)
        else:
            self.cri_gau_lap_G = None


        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # control how much emphasis is put on L(G(z_D)) during gradient descent.
        self.k = train_opt['init_k']
        self.M_global = AverageMeter()
        self.gamma = train_opt['gamma']
        self.lambda_k = train_opt['lambda_k']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(),
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                **train_opt['optim_d'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True
        l_d_total = 0
        l_d_real = 0
        l_d_fake = 0

        self.optimizer_d.zero_grad()

        recon_real = self.net_d(self.gt)
        self.output = self.net_g(self.lq)
        recon_fake = self.net_d(self.output.detach())

        # BEGAN loss
        l_d_began_real = self.cri_gan(recon_real, self.gt)
        l_d_began_fake = self.cri_gan(recon_fake, self.output)
        l_d_real += l_d_began_real
        l_d_fake += l_d_began_fake

        # wavelet loss
        if self.wavelet_weight_D > 0:
            wavelet_recon_real = self.wavelet_dec(recon_real)
            wavelet_recon_fake = self.wavelet_dec(recon_fake)
            wavelet_target = self.wavelet_dec(self.gt)
            wavelet_fake = self.wavelet_dec(self.output)
            l_d_wave_real = self.cri_wave_D(wavelet_recon_real, wavelet_target)
            l_d_wave_fake = self.cri_wave_D(wavelet_recon_fake, wavelet_fake)
            l_d_text_real = self.cri_text_D(wavelet_recon_real[:, 0:3, :, :], wavelet_target[:, 0:3, :, :])
            l_d_text_fake = self.cri_text_D(wavelet_recon_fake[:, 0:3, :, :], wavelet_fake[:, 0:3, :, :])
            l_d_real += self.wavelet_weight_D * l_d_wave_real + self.texture_weight_D * l_d_text_real
            l_d_fake += self.wavelet_weight_D * l_d_wave_fake + self.texture_weight_D * l_d_text_fake

        # gau_lap loss
        if self.gau_lap_weight_D > 0:
            Gau_Lap_recon_fake = self.Laplacian(self.GaussianBlur(recon_fake))
            Gau_Lap_recon_real = self.Laplacian(self.GaussianBlur(recon_real))
            Gau_Lap_target = self.Laplacian(self.GaussianBlur(self.gt))
            Gau_Lap_fake = self.Laplacian(self.GaussianBlur(self.output))
            l_d_lap_real = self.cri_gau_lap_D(Gau_Lap_recon_real, Gau_Lap_target)
            l_d_lap_fake = self.cri_gau_lap_D(Gau_Lap_recon_fake, Gau_Lap_fake)
            l_d_real += self.gau_lap_weight_D * l_d_lap_real
            l_d_fake += self.gau_lap_weight_D * l_d_lap_fake

        l_d_total = l_d_real - self.k * l_d_fake
        l_d_total.backward(retain_graph=True)
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        #loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        #loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

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

            # lpips loss
            if self.model_LPIPS:
                loss_LPIPS, _ = self.model_LPIPS.forward_pair(self.gt * 2 - 1, self.output * 2 - 1)
                loss_LPIPS = torch.mean(loss_LPIPS) * self.lpips_weight
                l_g_total += loss_LPIPS
                loss_dict['l_lpips'] = loss_LPIPS

            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_g_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_g_total += l_style
                    loss_dict['l_style'] = l_style

            # gan loss
            recon_fake = self.net_d(self.output)
            l_g_began = self.cri_gan(recon_fake, self.output)
            l_g_total += l_g_began
            loss_dict['l_g_gan'] = l_g_began

            # G wavelet loss
            if self.wavelet_weight_G > 0:
                wavelet_recon_fake = self.wavelet_dec(recon_fake)
                wavelet_fake = self.wavelet_dec(self.output)
                l_g_wavelet_fake = self.cri_wave_G(wavelet_fake, wavelet_recon_fake)
                l_g_texture_fake = self.cri_text_G(wavelet_fake, wavelet_recon_fake)
                l_g_total += self.wavelet_weight_G * l_g_wavelet_fake + self.texture_weight_G * l_g_texture_fake

            # G gau_lap loss
            if self.gau_lap_weight_G > 0:
                Gau_Lap_recon_fake = self.Laplacian(self.GaussianBlur(recon_fake))
                Gau_Lap_fake = self.Laplacian(self.GaussianBlur(self.output))
                l_g_Gau_Lap_fake = self.cri_gau_lap_G(Gau_Lap_recon_fake, Gau_Lap_fake)
                l_g_total += self.gau_lap_weight_G * l_g_Gau_Lap_fake

            loss_dict['l_g_total'] = l_g_total

            l_g_total.backward()
            self.optimizer_g.step()

        balance = (self.opt['train']["gamma"] * l_d_real - l_d_fake).item()
        self.k = min(max(self.k + self.opt['train']['lambda_k'] * balance, 0), 1)
        measure = l_d_real.item() + np.abs(balance)
        self.M_global.update(measure, self.gt.size(0))
        loss_dict['balance'] = balance
        loss_dict['k'] = self.k
        loss_dict['M_global'] = self.M_global.avg

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                if isinstance(value, int) or isinstance(value, float):
                    log_dict[name] = value
                else:
                    log_dict[name] = value.mean().item()

            return log_dict

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
