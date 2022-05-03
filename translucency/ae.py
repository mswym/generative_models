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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
import matplotlib.pyplot as plt
import copy
import numpy as np
from pytorch_lightning import loggers as pl_loggers
from class_mydataset_wo_openexr import MyDatasetBinary

class AE(LightningModule):
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
            input_height: int = 256,
            input_channels: int = 3,
            hidden_dims: list = None,
            enc_type: str = "resnet18",
            first_conv: bool = False,
            maxpool1: bool = False,
            enc_out_height: int = 16,
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
            hidden_dims = [128, 256, 512, 1024]
        self.enc_out_height = enc_out_height
        self.enc_out_dim = hidden_dims[-1]
        self.hidden_dims = hidden_dims
        self.input_channels = input_channels

        self.encoder = self.build_encoder(self.input_channels)
        self.fc_mu = nn.Linear(hidden_dims[-1]*enc_out_height*enc_out_height, self.latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*enc_out_height*enc_out_height)
        self.hidden_dims.reverse()
        self.decoder = self.build_decoder()
        self.final_layer = self.build_decoder_finallayer()
        self.hidden_dims.reverse()

    def build_encoder(self, input_channels):
        # Build Encoder
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            input_channels = h_dim

        return nn.Sequential(*modules)

    def build_decoder(self):
        modules = []
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        return nn.Sequential(*modules)

    def build_decoder_finallayer(self):
        module = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dims[-1],
                               self.hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
                nn.BatchNorm2d(self.hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(self.hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
                nn.Tanh())
        return module

    def decoders(self,z):
        x = self.decoder_input(z)
        x = x.view(-1, self.enc_out_dim, self.enc_out_height, self.enc_out_height)
        x = self.decoder(x)
        return self.final_layer(x)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) # due to manual encoder
        z = self.fc_mu(x)
        return self.decoders(z)

    def _run_step(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) # due to manual encoder
        z = self.fc_mu(x)
        return z, self.decoders(z)

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat     = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        loss = recon_loss

        logs = {
            "recon_loss": recon_loss,
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

def batch_run_ae(latent_dims):
    num_epochs = 100

    batch_size = 16
    learning_rate = 1e-5
    size_input = np.array([256, 256, 3])
    ratio_trainval = 0.9
    kl_coeff = 0.00001

    #list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    list_objname = ['bust', 'bunny', 'sphere', 'armadillo', 'buddha', 'bun','cap', 'cube', 'dragon', 'lucy', 'star_smooth']
    #path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap_mask/'
    path_dir_save = '/home/mswym/workspace/db/model_objects_tonemap_mask/'

    #latent_dims = [16]

    for latent_dim in latent_dims:
        for ind_obj in list_objname:
            log = []
            mypath = path_dir_save + 'che_220322_1500train_' + ind_obj + '.binary'
            tb_logger = pl_loggers.TensorBoardLogger(
                save_dir=path_dir_save,
                name='infovae_' + ind_obj + '_latent' + str(latent_dim) + "_logs/")

            #load mean information and make transforms
            #mean_img, std_img = load_mean_std(mypath)
            img_transform = transforms.Compose([
                transforms.Resize((size_input[0], size_input[1])),
                transforms.ToTensor()
            ])
            #load dataset while normalizing
            dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)

            train_data, val_data = random_split(dataset, [int(len(dataset) * ratio_trainval),
                                                          int(len(dataset) - len(dataset) * ratio_trainval)])
            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


            model = AE(input_height=size_input[0], input_channels=3, kl_coeff=kl_coeff, latent_dim=latent_dim,
                               lr=learning_rate, val_losses=log)
            trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, logger=tb_logger)
            trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    num_epochs = 100

    batch_size = 16
    learning_rate = 1e-5
    size_input = np.array([256, 256, 3])
    ratio_trainval = 0.9
    kl_coeff = 0.00001

    #list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    list_objname = ['bust', 'bunny', 'sphere', 'armadillo', 'buddha', 'bun','cap', 'cube', 'dragon', 'lucy', 'star_smooth']
    #path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap_mask/'
    path_dir_save = '/home/mswym/workspace/db/model_objects_tonemap_mask/'

    latent_dims = [16]

    for latent_dim in latent_dims:
        for ind_obj in list_objname:
            log = []
            mypath = path_dir_save + 'che_220322_1500train_' + ind_obj + '.binary'
            tb_logger = pl_loggers.TensorBoardLogger(
                save_dir=path_dir_save,
                name='ae_' + ind_obj + '_latent' + str(latent_dim) + "_logs/")

            #load mean information and make transforms
            #mean_img, std_img = load_mean_std(mypath)
            img_transform = transforms.Compose([
                transforms.Resize((size_input[0], size_input[1])),
                transforms.ToTensor()
            ])
            #load dataset while normalizing
            dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)

            train_data, val_data = random_split(dataset, [int(len(dataset) * ratio_trainval),
                                                          int(len(dataset) - len(dataset) * ratio_trainval)])
            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


            model = AE(input_height=size_input[0], input_channels=3, kl_coeff=kl_coeff, latent_dim=latent_dim,
                               lr=learning_rate, val_losses=log)
            trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, logger=tb_logger)
            trainer.fit(model, train_dataloader, val_dataloader)
