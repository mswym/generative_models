import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from class_mydataset import MyDataset_fromdir,MyDataset_frombinary


import numpy as np
import matplotlib.pyplot as

if __name__ == '__main__':

    fname_dir = './vae_img'
    num_epochs = 200
    batch_size = 300
    learning_rate = 1e-3
    size_input = np.array([256,256,1])


    #list_objname = ['a0.39','a0.59','a0.74','a0.8  7','a0.95']
    #list_objname = ['d36','d44','d54','d67','d82','d100','d122','d150','d184','d225','d276']
    #list_objname = ['d276']
    list_objname = ['armadillo','buddha','bun','bunny','bust','cap','cube','dragon','lucy','star_smooth']
    path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_2022/model_objects_notnormalize/'

    for ind_obj in range(len(list_objname)):
        mypath = path_dir_save+'che_03112021_1500train_'+list_objname[ind_obj] +'.binary'
        fname_save = path_dir_save+'sim_vae_batch300_notnormalize_'+list_objname[ind_obj]+'.pth'

        img_transform = transforms.Compose([
            transforms.Resize([size_input[0], size_input[1]]),
            transforms.ToTensor()
        ])

        val_dataset = MyDataset_frombinary(mypath, transform1=img_transform, flag_hdr=True)
        dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        model = VAE(size_input=size_input).cuda()
        BCE = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        list_loss = []
        for epoch in range(num_epochs):
            for data in dataloader:
                img, _ = data