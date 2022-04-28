# from https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split

from pytorch_lightning import loggers as pl_loggers
from utils import *

from class_mydataset_wo_openexr import MyDatasetBinary

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()

        self.hidden_dims = [128, 256, 512, 1024]
        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] * 16 * 16)
        self.hidden_dims.reverse()
        self.decoder = self.build_decoder()
        self.final_layer = self.build_decoder_finallayer()
        self.hidden_dims.reverse()
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

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[-1], 16, 16)
        x = self.decoder(x)
        return self.final_layer(x)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity



class GAN(LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 16,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []



class Encoder(LightningModule):
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

    def __init__(
            self,
            model,
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
        self.fc_var = nn.Linear(hidden_dims[-1]*enc_out_height*enc_out_height, self.latent_dim)

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


    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) # due to manual encoder
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z

    def _run_step(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) # due to manual encoder
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, p, q = self._run_step(x)
        x_hat = model.generator(z)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

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


if __name__ == '__main__':
    num_epochs = 100

    batch_size = 16
    learning_rate = 1e-4
    size_input = np.array([256, 256, 3])
    ratio_trainval = 0.9
    kl_coeff = 0.00001

    #list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    list_objname = ['bust', 'bunny', 'buddha']
    #path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap_mask/'
    path_dir_save = '/home/mswym/workspace/db/model_objects_tonemap_mask/'

    latent_dims = [16]

    for latent_dim in latent_dims:
        for ind_obj in list_objname:
            log = []
            mypath = path_dir_save + 'che_220322_1500train_' + ind_obj + '.binary'
            tb_logger_gan = pl_loggers.TensorBoardLogger(
                save_dir=path_dir_save,
                name='gan_' + ind_obj + '_latent' + str(latent_dim) + "_logs/")

            tb_logger_encode = pl_loggers.TensorBoardLogger(
                save_dir=path_dir_save,
                name='ganencode_' + ind_obj + '_latent' + str(latent_dim) + "_logs/")

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

            model = GAN(channels=size_input[2],
                        width=size_input[0],
                        height=size_input[1],
                        latent_dim=latent_dim,
                        lr=learning_rate,
                        b1=0.5,
                        b2=0.999,
                        batch_size= batch_size)
            trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, logger=tb_logger_gan)
            trainer.fit(model, train_dataloader, val_dataloader)


            model_encode = Encoder(model, input_height=size_input[0],
                                   input_channels=3,
                                   kl_coeff=kl_coeff,
                                   latent_dim=latent_dim,
                                   lr=learning_rate)
            trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, logger=tb_logger_encode)
            trainer.fit(model_encode, train_dataloader, val_dataloader)
