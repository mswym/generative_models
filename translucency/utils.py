import pytorch_lightning as pl
from vae_vanilla_resnet import VAE_resnet
from vae_vanilla import VAE_vanilla
from class_mydataset import MyDatasetDir,MyDatasetBinary
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

def check_output(model,img,ind_img):
    model.eval()
    with torch.no_grad():
        img_hat = model.forward(img)
        recon_loss = F.mse_loss(img_hat[ind_img], img[ind_img], reduction="mean")
        print(recon_loss)

        img_hat = img_hat.to('cpu').detach().numpy().transpose(0,2,3,1).copy()
        plt.imshow(img_hat[ind_img])
        plt.show()

        img = img.to('cpu').detach().numpy().transpose(0,2,3,1).copy()
        plt.imshow(img[ind_img])
        plt.show()

if __name__ == '__main__':
    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/'
    path_checkpoint = "/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/armadillo_latent20_logs/version_6/checkpoints/epoch=101-step=1427.ckpt"
    ind_obj = 0
    ind_img = 80

    num_epochs = 300
    batch_size = 100
    learning_rate = 1e-4
    size_input = np.array([256, 256, 3])
    ratio_trainval = 0.9
    latent_dim = 10
    log = []


    img_transform = transforms.Compose([
        transforms.Resize((size_input[0], size_input[1])),
        transforms.ToTensor()
    ])
    latent_dim = 20

    mypath = path_dir_save+'che_220322_1500train_'+list_objname[ind_obj] +'.binary'

    dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data = next(iter(test_dataloader))
    img = data[0]

    model = VAE_vanilla.load_from_checkpoint(checkpoint_path=path_checkpoint)
    check_output(model, img, ind_img)