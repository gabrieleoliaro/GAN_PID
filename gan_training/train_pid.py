# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
from gan_training.random_queue import Random_queue


class Trainer(object):
    def __init__(self,
                 generator,
                 discriminator,
                 g_optimizer,
                 d_optimizer,
                 gan_type,
                 reg_type,
                 reg_param,
                 pv=1,
                 iv=0,
                 dv=0,
                 time_step=1.,
                 batch_size=64,
                 config=None):
        print("Using PID Trainer")
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param

        self.d_xfake = None
        self.d_previous_z = None
        self.d_previous_y = None

        self.pv = pv
        self.iv = iv
        self.dv = dv
        self.time_step = time_step
        self.batch_size = batch_size
        self.config = config
        self.i_real_queue = Random_queue(
            config['training']['batch_size'] *
            config['training']['i_buffer_factor'],
            config['training']['batch_size'])

        self.i_fake_queue = Random_queue(
            config['training']['batch_size'] *
            config['training']['i_buffer_factor'],
            config['training']['batch_size'])

        self.max0 = torch.nn.ReLU()

    def generator_trainstep(self, y, z):
        if self.config["generator"]["name"] != "wgan":
            assert (y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()
        if self.config["generator"]["name"] != "wgan":
            # Feed z (a minimibatch of noise samples from the gaussian distro), together with y (image class labels) to the generator. The generator will output fake samples, i.e. x_fake are fake images!
            x_fake = self.generator(z, y)
            # classify generator output (x_fake) with the discriminator
            d_fake = self.discriminator(x_fake, y)
        else:
            # Feed z (a minimibatch of noise samples from the gaussian distro), together with y (image class labels) to the generator. The generator will output fake samples, i.e. x_fake are fake images!
            x_fake = self.generator(z)
            # classify generator output (x_fake) with the discriminator
            d_fake = self.discriminator(x_fake)
        # compute G's loss using real labels as GT
        gloss = self.compute_loss(d_fake, 1, is_generator=True)
        # compute G's gradients in backward pass
        gloss.backward()
        # update G's parameters with an optimizer's step
        self.g_optimizer.step()

        return gloss.item()

    def discriminator_trainstep(self, x_real, y, z, it=0):
        # print(it)
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # we are applying PID control using three coefficients:
        # - pv for the proportional control
        # - iv for the integral control
        # - dv for the derivative control

        reg_d = self.config['training']['regularize_output_d'] # unused in CIFAR
        # classify real image(s) (x_real) with the discriminator
        if self.config["discriminator"]["name"] != "wgan":
            d_real = self.discriminator(x_real, y)
        else:
            d_real = self.discriminator(x_real)
        # compute D's loss using real labels as GT
        dloss_real = self.compute_loss(d_real, 1) * self.pv # P in PID applied here
        
        # unused in CIFAR, no need to worry
        if reg_d > 0.:
            dloss_real += (d_real**2).mean() * reg_d
        # compute D's gradients in backward pass
        dloss_real.backward()

        # On fake data
        # Feed a minibatch of noise samples to the generator, to generate a fake sample from the given class
        with torch.no_grad():
            if self.config["discriminator"]["name"] != "wgan":
                x_fake = self.generator(z, y) # apply the generator to obtain a fake image from the class we are working on. I guess this should be the only way to call the generator, right? Because it wouldn't make sense to call the generator on real data.
            else:
                x_fake = self.generator(z)
        # classify generator output (x_fake) with the discriminator
        if self.config["discriminator"]["name"] != "wgan":
            d_fake = self.discriminator(x_fake, y)
        else:
            d_fake = self.discriminator(x_fake)
        # compute G's loss using real labels as GT, then multiply by the P proportional coefficient
        dloss_fake = self.compute_loss(d_fake, 0) * self.pv
        if reg_d > 0.:
            dloss_fake += (d_fake**2).mean() * reg_d
        # compute G's gradients in backward pass
        dloss_fake.backward()

        i_loss = torch.from_numpy(np.array([0.]))
        if self.iv > 0:
            # i_factor = self.config['training']['i_buffer_factor']
            # i_store = self.config['training']['i_buffer_onestep']
            if self.config["discriminator"]["name"] != "wgan":
                xtmp = x_real.detach().cpu().numpy()
                ytmp = y.detach().cpu().numpy()
                self.i_real_queue.set_data(xtmp, ytmp)

                xtmp = x_fake.detach().cpu().numpy()
                ytmp = y.detach().cpu().numpy()
                self.i_fake_queue.set_data(xtmp, ytmp)

                i_xreal, i_yreal = self.i_real_queue.get_data()
                i_xfake, i_yfake = self.i_fake_queue.get_data()

                i_xreal = torch.as_tensor(i_xreal, dtype=torch.float32).cuda()
                i_xfake = torch.as_tensor(i_xfake, dtype=torch.float32).cuda()
                i_yreal = torch.as_tensor(i_yreal, dtype=torch.long).cuda()
                i_yfake = torch.as_tensor(i_yfake, dtype=torch.long).cuda()
            else:
                xtmp = x_real.detach().cpu().numpy()
                self.i_real_queue.set_data(xtmp)
                xtmp = x_fake.detach().cpu().numpy()
                self.i_fake_queue.set_data(xtmp)

                i_xreal = self.i_real_queue.get_data()
                i_xfake = self.i_fake_queue.get_data()

                i_xreal = torch.as_tensor(i_xreal, dtype=torch.float32).cuda()
                i_xfake = torch.as_tensor(i_xfake, dtype=torch.float32).cuda()
                

            if self.config["discriminator"]["name"] != "wgan":
                i_real_doutput = self.discriminator(i_xreal, i_yreal)
                i_loss_real = self.compute_loss(i_real_doutput, 1)
                i_fake_doutput = self.discriminator(i_xfake, i_yfake)
                i_loss_fake = self.compute_loss(i_fake_doutput, 0)
            else:
                i_real_doutput = self.discriminator(i_xreal)
                i_loss_real = self.compute_loss(i_real_doutput, 1)
                i_fake_doutput = self.discriminator(i_xfake)
                i_loss_fake = self.compute_loss(i_fake_doutput, 0)

            if self.config['training']['pid_type'] == 'function':
                i_loss = (i_loss_real + i_loss_fake) * self.iv
            elif self.config['training']['pid_type'] == 'square':
                i_loss = ((i_real_doutput**2).mean() +
                          (i_fake_doutput**2).mean()) * self.iv
            elif self.config['training']['pid_type'] == 'abs':
                i_loss = (torch.abs(i_real_doutput).mean() +
                          torch.abs(i_fake_doutput).mean()) * self.iv
            elif self.config['training']['pid_type'] == 'accurate':
                i_fake_doutput = self.max0(i_fake_doutput)
                i_real_doutput = -1 * self.max0(-1 * i_real_doutput)
                i_loss = (i_fake_doutput - i_real_doutput).mean() * self.iv
            i_loss.backward()

        d_loss = torch.from_numpy(np.array([0.]))
        # print(self.dv)
        if self.dv > 0 and it > 0:
            if self.d_xfake is None:
                self.d_xreal = x_real
                self.d_xfake = x_fake
                self.d_previous_z = z
                if self.config["discriminator"]["name"] != "wgan":
                    self.d_previous_y = y
            else:
                if self.config["discriminator"]["name"] != "wgan":
                    d_loss_previous_f = self.compute_loss(
                        self.discriminator(self.d_xfake, self.d_previous_y), 0)
                    d_loss_previous_r = self.compute_loss(
                        self.discriminator(self.d_xreal, self.d_previous_y), 1)
                else:
                    d_loss_previous_f = self.compute_loss(
                        self.discriminator(self.d_xfake), 0)
                    d_loss_previous_r = self.compute_loss(
                        self.discriminator(self.d_xreal), 1)
                d_loss_previous = d_loss_previous_f + d_loss_previous_r

                d_loss_current_f = self.compute_loss(
                    self.discriminator(x_fake, y), 0)
                d_loss_current_r = self.compute_loss(
                    self.discriminator(x_real, y), 1)
                d_loss_current = d_loss_current_f + d_loss_current_r

                d_loss = (d_loss_current - d_loss_previous) * self.dv
                d_loss.backward()

                self.d_xreal = x_real
                self.d_xfake = x_fake
                self.d_previous_z = z
                if self.config["discriminator"]["name"] != "wgan":
                    self.d_previous_y = y

        self.d_optimizer.step()
        toggle_grad(self.discriminator, False)

        # Output
        dloss = (dloss_real + dloss_fake)
        return dloss.item(), d_loss.item(), i_loss.item()

    def compute_loss(self, d_out, target, is_generator=False):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2 * target - 1) * d_out.mean()
        elif self.gan_type == 'hinge':
            if is_generator is False:
                loss = (F.relu(1 + (2 * target - 1) * d_out)).mean()
            else:
                loss = ((2 * target - 1) * d_out).mean()
        elif self.gan_type == 'sigmoid':
            d_out = d_out * self.config['training']['sigmoid_coe']
            loss = ((2 * target - 1) * F.sigmoid(d_out)
                    ).mean() / self.config['training']['sigmoid_coe']
        elif self.gan_type == 'lsgan1':
            if is_generator is False:
                target = target * 2 - 1
                loss = ((d_out - target)**2).mean()
            else:
                loss = (d_out**2).mean()
        elif self.gan_type == 'lsgan2':
            target -= 0.5
            loss = ((d_out - target)**2).mean()
        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(outputs=d_out.sum(),
                              inputs=x_in,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)
