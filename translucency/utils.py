import pytorch_lightning as pl
from vae_vanilla_resnet import VAE
from class_mydataset import MyDatasetDir,MyDatasetBinary
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import torch

def check_output(model,img,ind_img):
    model.eval()
    with torch.no_grad():
        img_hat = model.forward(img)
        img_hat = img_hat.to('cpu').detach().numpy().transpose(0,2,3,1).copy()
        plt.imshow(img_hat[ind_img])
        plt.show()

if __name__ == '__main__':
    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth','sphere']
    path_dir_save = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/'
    ind_obj = 0
    size_input = np.array([128, 128, 3])
    ratio_trainval = 0.95
    batch_size = 300
    ind_img = 0
    img_transform = transforms.Compose([
        transforms.Resize((size_input[0], size_input[1])),
        transforms.ToTensor()
    ])

    mypath = path_dir_save+'che_220322_300test_'+list_objname[ind_obj] +'.binary'

    dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data = [tmp for tmp in test_dataloader]
    img,_ = data[0]

    model = VAE.load_from_checkpoint("/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap/armadillo/lightning_logs/version_11/checkpoints/epoch=250-step=11294.ckpt")
    check_output(model, img, ind_img)