# this code is edited from pytorch lightning code.
# https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py

# I also added/edited the codes from AntixK/Pytorch-VAE repository.
# https://github.com/AntixK/PyTorch-VAE

import urllib.parse
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import nn
from torch.nn import functional as F

from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
import matplotlib.pyplot as plt
import copy
from torch.autograd import Variable


class VAE_noncnn(LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
        # pretrained on stl10
        vae = VAE(input_height=32).from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        "cifar10-resnet18": urllib.parse.urljoin(_HTTPS_AWS_HUB, "vae/vae-cifar10/checkpoints/epoch%3D89.ckpt"),
        "stl10-resnet18": urllib.parse.urljoin(_HTTPS_AWS_HUB, "vae/vae-stl10/checkpoints/epoch%3D89.ckpt"),
    }

    def __init__(
            self,
            input_height: int,
            input_channels: int,
            hidden_dims: list = None,
            kl_coeff: float = 0.1,
            latent_dim: int = 256,
            lr: float = 1e-4,
            val_losses: list = None,
            **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.val_losses = val_losses

        if hidden_dims is None:
            hidden_dims = [400]
        self.input_channels = input_channels

        self.encoder = nn.Sequential(
            nn.Linear(self.input_height * self.input_height * input_channels, hidden_dims[0]),
            nn.ReLU(True))

        self.fc_mu = nn.Linear(hidden_dims[0], self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[0], self.latent_dim)


        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dims[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dims[0], self.input_height * self.input_height * input_channels),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) # due to manual encoder
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat

    def _run_step(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) # due to manual encoder
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = Variable(torch.cuda.FloatTensor(std.size()).normal_())
        z.mul(std).add(mu)

        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, self.input_height * self.input_height * self.input_channels)
        z, x_hat, p, q = self._run_step(x)

        #recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()},
                      prog_bar=True,
                      logger=True,
                      on_step=True,
                      on_epoch=True)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.val_losses.append(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
