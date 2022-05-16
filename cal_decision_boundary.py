import os
import numpy as np
import matplotlib.pyplot as plt
import time

from translucency.utils import *
from translucency.class_mydataset import MyDatasetBinary
from translucency.vae_vanilla import VAE_vanilla

from sklearn import svm
import itertools

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


def cal_svm(X, label):
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X, label)

    return svm_model


def extract_latent_xhat(model, img_train):
    latent_z, img_hat, _, _ = model._run_step(img_train)

    return latent_z.to('cpu').detach().numpy().copy(), img_hat.to('cpu').detach().numpy().copy()

def extract_latents_xhats_imgs(num_itr, model, img_train):
    model.eval()
    latent_z = []
    x_hat = []
    for itr in range(num_itr):
        tmp_z, tmp_xhat = extract_latent_xhat(model, next(iter(img_train)))
        latent_z.append(tmp_z)
        x_hat.append(tmp_xhat)

    latent_z = np.array(latent_z)
    x_hat = np.array(x_hat)
    return latent_z.reshape([-1, latent_z.shape[2]]), x_hat.reshape(
        [-1, x_hat.shape[2], x_hat.shape[3], x_hat.shape[4]])


if __name__ == '__main__':
    size_input = np.array([256, 256, 3])
    scale_eigenval = 1
    flag_autoadjust_eigenval = False

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
                    'sphere']
    # list_num_latent = [16]
    path_dir_save = '/media/mswym/PortableSSD/translucency/'

    model_body = VAE_vanilla()
    num_latent = 16
    #ind_obj_1 = 'lucy'
    #ind_obj_2 = 'bun'

    comb_list = list(itertools.combinations(np.linspace(0, len(list_objname) - 1, len(list_objname)), 2))

    for ind_obj in range(len(comb_list)):
        start_time = time.perf_counter()
        ind_obj_1 = list_objname[int(comb_list[ind_obj][0])]
        ind_obj_2 = list_objname[int(comb_list[ind_obj][1])]

        path_save_1 = path_dir_save + "obj_mask/svm/svm_" + ind_obj_1 + "-" + ind_obj_2 + "_latent" + str(
            num_latent) + ".pkl"
        path_save_2 = path_dir_save + "obj_mask/svm/svm_" + ind_obj_2 + "-" + ind_obj_1 + "_latent" + str(
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
        latent_z_1, img_hat_1 = extract_latents_xhats_imgs(len(img_train_1), model_1, img_train_1)
        latent_z_2, img_hat_2 = extract_latents_xhats_imgs(len(img_train_2), model_2, img_train_2)

        # take svm for the z values
        svm_model_1 = cal_svm(latent_z_1, np.array(labels_1))
        svm_model_2 = cal_svm(latent_z_2, np.array(labels_2))
        # save the svm models
        with open(path_save_1, 'wb') as f:
            pickle.dump(svm_model_1, f)
        with open(path_save_2, 'wb') as f:
            pickle.dump(svm_model_2, f)
        print(time.perf_counter() - start_time)
    # control the boundary values
