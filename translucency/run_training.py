import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from class_mydataset import MyDatasetDir, MyDatasetBinary

#from vae_vanilla_resnet import VAE_resnet
#from vae_vanilla import VAE_vanilla
from vae_noncnn import VAE_noncnn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    num_epochs = 200

    batch_size = 100
    learning_rate = 1e-3
    size_input = np.array([256, 256, 1])
    ratio_trainval = 0.9
    latent_dim = 20

    #list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
    #                'sphere']
    list_objname = ['armadillo']
    #path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/'
    path_dir_save = '../'


    img_transform = transforms.Compose([
        transforms.Resize((size_input[0], size_input[1])),
        transforms.ToTensor()
    ])

    for ind_obj in list_objname:
        log = []
        mypath = path_dir_save + 'che_220322_1500train_' + ind_obj + '.binary'
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=path_dir_save,
            name=ind_obj + '_latent' + str(latent_dim) + "_logs/")

        dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)
        train_data, val_data = random_split(dataset, [int(len(dataset) * ratio_trainval),
                                                      int(len(dataset) - len(dataset) * ratio_trainval)])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        model = VAE_noncnn(input_height=size_input[0], input_channels=1, latent_dim=latent_dim, lr=learning_rate, val_losses=log)
        trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, logger=tb_logger)
        trainer.fit(model, train_dataloader, val_dataloader)

        # save loss list (though pytorch lightning makes the log for torchboard)
        log = [tmp.to('cpu').detach().numpy().copy() for tmp in log]
        open_file = open(path_dir_save + ind_obj + '_latent' + str(latent_dim) + "_logs/" + '/loss_logs.pkl', "wb")
        pickle.dump(log, open_file)
        open_file.close()
