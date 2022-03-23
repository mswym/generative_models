import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
from class_mydataset import MyDatasetDir, MyDatasetBinary

from vae_vanilla_resnet import VAE
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    num_epochs = 300
    batch_size = 10
    learning_rate = 1e-4
    size_input = np.array([256, 256, 3])
    ratio_trainval = 0.95
    latent_dim = 10

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
                    'sphere']
    path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/'

    tb_logger = pl_loggers.TensorBoardLogger("logs/")

    img_transform = transforms.Compose([
        transforms.Resize((size_input[0], size_input[1])),
        transforms.ToTensor()
    ])

    for ind_obj in list_objname:
        log = []
        mypath = path_dir_save + 'che_220322_1500train_' + ind_obj + '.binary'
        fname_save = path_dir_save + 'sim_vae_batch300_' + ind_obj + '.pth'

        dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)
        train_data, val_data = random_split(dataset, [int(len(dataset) * ratio_trainval),
                                                      int(len(dataset) - len(dataset) * ratio_trainval)])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        model = VAE(input_height=size_input[0], latent_dim=latent_dim, lr=learning_rate, val_losses=log)
        trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, default_root_dir=path_dir_save + ind_obj, logger=tb_logger)
        trainer.fit(model, train_dataloader, val_dataloader)

        # save loss list
        open_file = open(path_dir_save + list_objname[ind_obj] + '/loss_logs.pkl', "wb")
        pickle.dump(log, open_file)
        open_file.close()
