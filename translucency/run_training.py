import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
from class_mydataset import MyDatasetDir,MyDatasetBinary


from vae_vanilla_resnet import VAE
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    fname_dir = './vae_img'
    num_epochs = 300
    batch_size = 10
    learning_rate = 1e-4
    size_input = np.array([256,256,1])
    ratio_trainval = 0.95
    latent_dim = 10


    list_objname = ['armadillo','buddha','bun','bunny','bust','cap','cube','dragon','lucy','star_smooth']
    path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/'

    img_transform = transforms.Compose([
        transforms.Resize([size_input[0], size_input[1]]),
        transforms.ToTensor()
    ])


    for ind_obj in range(len(list_objname)):
        mypath = path_dir_save+'che_220322_1500train_'+list_objname[ind_obj] +'.binary'
        fname_save = path_dir_save+'sim_vae_batch300_'+list_objname[ind_obj]+'.pth'

        dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)
        train_data, val_data = random_split(dataset, [int(len(dataset)*ratio_trainval), int(len(dataset)-len(dataset)*ratio_trainval)])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        model = VAE(input_height=size_input[0] * size_input[1] * size_input[2], latent_dim=latent_dim, lr=learning_rate)
        trainer = pl.Trainer(gpus=1, max_epochs=num_epochs,default_root_dir=path_dir_save+list_objname[ind_obj])
        trainer.fit(model, train_dataloader, val_dataloader)
