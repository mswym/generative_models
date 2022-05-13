import os
import numpy as np
import matplotlib.pyplot as plt
import time

from translucency.utils import *
from translucency.class_mydataset import MyDatasetBinary
from translucency.vae_vanilla import VAE_vanilla


def extract_opaque_trans(dataloader):
    imgs = []
    labels = []
    for data in dataloader:
        img, label = data
        if label[0][2] == 276: #opaque
            imgs.append(img)
            labels.append(0)
        elif label[0][2] == 36: #translucent
            imgs.append(img)
            labels.append(1)


    return imgs, labels


if __name__ == '__main__':
    size_input = np.array([256, 256, 3])
    scale_eigenval = 1
    flag_autoadjust_eigenval = False

    # list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
    #                'sphere']
    # list_num_latent = [16]
    path_dir_save = '/media/mswym/PortableSSD/translucency/'

    model_body = VAE_vanilla()
    num_latent = 16
    ind_obj_1 = 'lucy'
    ind_obj_2 = 'bun'

    start_time = time.perf_counter()

    path_decoder_load_1 = path_dir_save + "obj_mask/coef/coef_" + ind_obj_1 + "-" + ind_obj_2 + "_latent" + str(
        num_latent) + ".pkl"
    path_decoder_load_2 = path_dir_save + "obj_mask/coef/coef_" + ind_obj_2 + "-" + ind_obj_1 + "_latent" + str(
        num_latent) + ".pkl"

    path_checkpoint_1 = path_dir_save + "obj_mask/" + ind_obj_1 + "_latent" + str(
        num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"
    path_checkpoint_2 = path_dir_save + "obj_mask/" + ind_obj_2 + "_latent" + str(
        num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"

    mypath_1 = path_dir_save + 'model_objects_tonemap_mask/che_220322_1500train_' + ind_obj_1 + '.binary'
    mypath_2 = path_dir_save + 'model_objects_tonemap_mask/che_220322_1500train_' + ind_obj_2 + '.binary'

    img_transform = transforms.Compose([
        transforms.Resize((size_input[0], size_input[1])),
        transforms.ToTensor()
    ])

    # load dataset
    img_train_1 = read_img_dataset(mypath_1, img_transform, 1)  # batche size 1
    img_train_2 = read_img_dataset(mypath_2, img_transform, 1)  # batche size 1

    # extract only opaque and translucenct images
    img_train_1, labels_1 = extract_opaque_trans(img_train_1)
    img_train_2, labels_2 = extract_opaque_trans(img_train_2)

    # calculate z value
    model_1 = model_body.load_from_checkpoint(checkpoint_path=path_checkpoint_1)
    model_2 = model_body.load_from_checkpoint(checkpoint_path=path_checkpoint_2)
    latent_z_1, img_hat_1 = extract_latents_xhats(len(img_train_1), model_1, img_train_1)
    latent_z_2, img_hat_2 = extract_latents_xhats(len(img_train_2), model_2, img_train_2)

    # take svm for the z values

    # save the svm models

    # control the boundary values