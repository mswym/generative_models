import pytorch_lightning as pl
from pl_bolts.datamodules import SklearnDataModule
from sklearn.datasets import load_boston
from translucency.linear_decoder import *

import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
#from vae_vanilla_resnet import VAE_resnet
from translucency.vae_vanilla import VAE_vanilla
##from vae_noncnn import VAE_noncnn
from translucency.class_mydataset import MyDatasetBinary
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import torch
from torch.nn import functional as F
import os
import pickle

# from sklearn.linear_model import LinearRegression


def cal_linear_reg_out_model(x, y, scale_eigenval = 1,autoadjust_eigenval=True):
    model_lr = LinearRegression(fit_intercept=True)
    model_lr.fit(x, y)
    eigen_val, eigen_vec = np.linalg.eig(model_lr.coef_)
    #eigen_val = eigen_val / np.linalg.norm(eigen_val)
    if autoadjust_eigenval:
        eigen_val = eigen_val / np.linalg.norm(eigen_val)
    else:
        eigen_val = eigen_val * scale_eigenval
    #eigen_val = np.ones(eigen_val.shape)
    eigen_val = np.diag(eigen_val)
    #for i in range(len(eigen_vec)):
    #    eigen_vec[i] = eigen_vec[i] / np.linalg.norm(eigen_vec[i])
    new_coef_matrix = np.dot(np.dot(eigen_vec, eigen_val), np.linalg.inv(eigen_vec))
    new_coef_matrix = np.real(new_coef_matrix)

    return new_coef_matrix, model_lr.intercept_


if __name__ == '__main__':
    batch_size_train = 1500
    batch_size_test = 300
    size_input = np.array([256, 256, 3])
    scale_eigenval = 1

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    path_dir_save = '../../db/obj_mask/'

    ind_obj_1 = 0
    ind_obj_2 = 1

    model_1 = VAE_vanilla()
    model_2 = VAE_vanilla()


    path_checkpoint_1 = "../../db/obj_mask/armadillo_latent256_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"
    path_checkpoint_2 = "../../db/obj_mask/buddha_latent256_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"

    mypath_1 = path_dir_save + '../model_objects_tonemap_mask/che_220322_1500train_' + list_objname[ind_obj_1] + '.binary'
    mypath_2 = path_dir_save + '../model_objects_tonemap_mask/che_220322_1500train_' + list_objname[ind_obj_2] + '.binary'
    img_transform = transforms.Compose([
        transforms.Resize((size_input[0], size_input[1])),
        transforms.ToTensor()
    ])
    dataset_1 = MyDatasetBinary(mypath_1, transform1=img_transform, flag_hdr=True)
    train_dataloader_1 = DataLoader(dataset_1, batch_size=batch_size_train, shuffle=False)
    img_train_1 = next(iter(train_dataloader_1))
    img_train_1 = img_train_1[0]

    dataset_2 = MyDatasetBinary(mypath_2, transform1=img_transform, flag_hdr=True)
    train_dataloader_2 = DataLoader(dataset_2, batch_size=batch_size_train, shuffle=False)
    img_train_2 = next(iter(train_dataloader_2))
    img_train_2 = img_train_2[0]


    #load trained models
    model_1 = model_1.load_from_checkpoint(checkpoint_path=path_checkpoint_1)
    model_1.eval()
    model_2 = model_2.load_from_checkpoint(checkpoint_path=path_checkpoint_2)
    model_2.eval()

    #get latent codes z
    latent_z_1, x_hat_1, _, _ = model_1._run_step(img_train_1)
    latent_z_2, x_hat_2, _, _ = model_2._run_step(img_train_2)

    latent_z_1 = latent_z_1.to('cpu').detach().numpy().copy()
    latent_z_2 = latent_z_2.to('cpu').detach().numpy().copy()

    #train the linear decoder
    coeff_1, intercept_1 = cal_linear_reg_out_model(latent_z_1, latent_z_2, scale_eigenval, autoadjust_eigenval=flag_autoadjust_eigenval)
    coeff_2, intercept_2 = cal_linear_reg_out_model(latent_z_2, latent_z_1, scale_eigenval, autoadjust_eigenval=flag_autoadjust_eigenval)

    a = 1


    print('finished')