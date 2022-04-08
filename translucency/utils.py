import pytorch_lightning as pl
#from vae_vanilla_resnet import VAE_resnet
from vae_vanilla import VAE_vanilla
#from vae_noncnn import VAE_noncnn
from class_mydataset import MyDatasetDir,MyDatasetBinary
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import os
import pickle


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
    path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/'
    #path_checkpoint = "/media/mswym/SSD-PGU3/database/results_translucen" \
    #                  "t_220303/model_objects_tonemap/armadillo_latent20_logs/version_25/checkpoints/epoch=18-step=265.ckpt"
    path_checkpoint = "/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/armadillo_latent20_logs/version_39/checkpoints/epoch=21-step=1869.ckpt"
    ind_obj = 0

    ind_img = 80

    num_epochs = 200
    batch_size = 100
    learning_rate = 1e-4
    size_input = np.array([256, 256, 3])
    ratio_trainval = 0.9
    latent_dim = 10
    log = []

    latent_dim = 100

    mypath = path_dir_save+'che_220322_1500train_'+list_objname[ind_obj] +'.binary'

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