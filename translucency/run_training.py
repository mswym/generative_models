import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from class_mydataset import MyDatasetDir, MyDatasetBinary

#from vae_vanilla_resnet import VAE_resnet
from vae_vanilla import VAE_vanilla
#from vae_noncnn import VAE_noncnn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils import *

if __name__ == '__main__':
    num_epochs = 150

    batch_size = 16
    learning_rate = 1e-5
    size_input = np.array([256, 256, 3])
    ratio_trainval = 0.9
    kl_coeff = 0.00001

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    #list_objname = ['armadillo']
    #path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap_mask/'
    path_dir_save = '../obj_mask_coef401/'

    latent_dims = [16, 256, 2, 4, 8, 32, 64, 128]

    for latent_dim in latent_dims:
        for ind_obj in list_objname:
            log = []
            mypath = path_dir_save + 'che_220322_1500train_' + ind_obj + '.binary'
            tb_logger = pl_loggers.TensorBoardLogger(
                save_dir=path_dir_save,
                name=ind_obj + '_latent' + str(latent_dim) + "_logs/")

            #load mean information and make transforms
            mean_img, std_img = load_mean_std(mypath)
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

            #model = VAE_noncnn(input_height=size_input[0], input_channels=3, kl_coeff=kl_coeff, latent_dim=latent_dim, lr=learning_rate, val_losses=log)
            model = VAE_vanilla(input_height=size_input[0], input_channels=3, kl_coeff=kl_coeff, latent_dim=latent_dim,
                               lr=learning_rate, val_losses=log)
            trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, logger=tb_logger)
            trainer.fit(model, train_dataloader, val_dataloader)

            # save loss list (though pytorch lightning makes the log for torchboard)
            log = [tmp.to('cpu').detach().numpy().copy() for tmp in log]
            open_file = open(path_dir_save + ind_obj + '_latent' + str(latent_dim) + "_logs/" + '/loss_logs.pkl', "wb")
            pickle.dump(log, open_file)
            open_file.close()
