import pytorch_lightning as pl
#from vae_vanilla_resnet import VAE_resnet
#from vae_vanilla import VAE_vanilla
#from vae_noncnn import VAE_noncnn
from translucency.class_mydataset_wo_openexr import MyDatasetBinary
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import os
import pickle


def read_img_dataset(mypath, img_transform, batch_size_train):
    # load all image data
    dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)
    dataset = DataLoader(dataset, batch_size=batch_size_train, shuffle=False)

    return dataset

def extract_latent_xhat(model, img_train):
    latent_z, img_hat, _, _ = model._run_step(img_train)
    #latent_z, img_hat = model._run_step(img_train)

    return latent_z.to('cpu').detach().numpy().copy(), img_hat.to('cpu').detach().numpy().copy()


def extract_latents_xhats(num_itr, model, img_train):
    model.eval()
    latent_z = []
    x_hat = []
    for data in img_train:
        img, _ = data
        tmp_z, tmp_xhat = extract_latent_xhat(model, img)
        latent_z.append(tmp_z)
        x_hat.append(tmp_xhat)

    latent_z = np.array(latent_z)
    x_hat = np.array(x_hat)
    return latent_z.reshape([-1, latent_z.shape[2]]), x_hat.reshape(
        [-1, x_hat.shape[2], x_hat.shape[3], x_hat.shape[4]])


def load_mean_std(mypath):
    # assuming 8bit input values and output 0-1 float values.
    tmp_dir, tmp_file = os.path.split(mypath)
    with open(tmp_dir + '/mean_' + tmp_file, 'rb') as f:
        mean_img = pickle.load(f)
    with open(tmp_dir + '/std_' + tmp_file, 'rb') as f:
        std_img = pickle.load(f)
    return mean_img/255, std_img/255

def check_output(model,img,ind_img):
    model.eval()
    #img = img.view(img.size()[0], -1)
    img_hat = model.forward(img)
    recon_loss = F.mse_loss(img_hat[ind_img], img[ind_img], reduction="mean")
    print(recon_loss)

    img = img.view(img.size()[0], 3, 256, 256)
    img_hat = img_hat.view(img_hat.size()[0], 3, 256, 256)

    img_panel_hat = torchvision.utils.make_grid(img_hat)
    img_panel_hat = img_panel_hat.to('cpu').detach().numpy().transpose(1,2,0).copy()
    img_panel = torchvision.utils.make_grid(img)
    img_panel = img_panel.to('cpu').detach().numpy().transpose(1,2,0).copy()


    img_hat = img_hat.to('cpu').detach().numpy().transpose(0,2,3,1).copy()
    plt.imshow(img_hat[ind_img], cmap=plt.get_cmap('gray'),vmin=0,vmax=1)
    plt.show()


    img = img.to('cpu').detach().numpy().\
        transpose(0,2,3,1).copy()
    plt.imshow(img[ind_img], cmap=plt.get_cmap('gray'),vmin=0,vmax=1)
    plt.show()
    print('finished')


if __name__ == '__main__':
    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    #path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/'
    path_dir_save = '../obj_mask/'
    #path_checkpoint = "/media/mswym/SSD-PGU3/database/results_translucen" \
    #                  "t_220303/model_objects_tonemap/armadillo_latent20_logs/version_25/checkpoints/epoch=18-step=265.ckpt"
    #path_checkpoint = "/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/armadillo_latent20_logs/version_39/checkpoints/epoch=21-step=1869.ckpt"
    path_checkpoint = "../obj_mask/armadillo_latent256_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"
    ind_obj = 0

    ind_img = 190


    num_epochs = 200
    batch_size = 300
    learning_rate = 1e-4
    size_input = np.array([256, 256, 3])
    ratio_trainval = 0.9
    latent_dim = 10
    log = []

    latent_dim = 128

    mypath = path_dir_save+'che_220322_300test_'+list_objname[ind_obj] +'.binary'
    mean_img, std_img = load_mean_std(mypath)
    img_transform = transforms.Compose([
        transforms.Resize((size_input[0], size_input[1])),
        transforms.ToTensor()
    ])

    dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data = next(iter(test_dataloader))
    img = data[0]
    model = VAE_vanilla.load_from_checkpoint(checkpoint_path=path_checkpoint)
    check_output(model, img, ind_img)