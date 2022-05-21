import os
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torchvision import transforms

from translucency.utils import *
from translucency.class_mydataset import MyDatasetBinary
from translucency.vae_vanilla import VAE_vanilla
from translucency.ae import AE

from sklearn.decomposition import PCA

import pickle


def extract_latent(model, img_train):
    latent_z, _, _, _ = model._run_step(img_train)
    #latent_z, _ = model._run_step(img_train)

    return latent_z.to('cpu').detach().numpy().copy()

def extract_latents(num_itr, model, path_checkpoint, img_train):
    model = model.load_from_checkpoint(checkpoint_path=path_checkpoint)
    model.eval()
    latent_z = []
    for data in img_train:
        img, _ = data
        latent_z.append(extract_latent(model, img))

    latent_z = np.array(latent_z)

    return latent_z.reshape([-1, latent_z.shape[2]])

def read_img_dataset(mypath, img_transform, batch_size_train):
    #load all image data
    dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)
    dataset = DataLoader(dataset, batch_size=batch_size_train, shuffle=False)

    return dataset

if __name__ == '__main__':
    batch_size_train = 10
    size_input = np.array([256, 256, 3])
    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
                    'sphere']

    path_dir_save = '/home/mswym/workspace/db/'
    path_save_sum = path_dir_save + 'vae_vanilla_screeplot.pickle'
    model_body = VAE_vanilla()
    list_num_latent = [128, 256]
    summary_explained_variance_ratio = []

    for num_latent in list_num_latent:
        explained_variance_ratio = []
        for ind_obj_1 in list_objname:
            print(ind_obj_1)

            start_time = time.perf_counter()

            path_checkpoint_1 = path_dir_save + "obj_mask/" + ind_obj_1 + "_latent" + str(
                num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"

            mypath_1 = path_dir_save + 'model_objects_tonemap_mask/che_220322_300test_' + ind_obj_1 + '.binary'


            img_transform = transforms.Compose([
                transforms.Resize((size_input[0], size_input[1])),
                transforms.ToTensor()
            ])

            # load dataset
            img_test_1 = read_img_dataset(mypath_1, img_transform, batch_size_train)

            # calculate z value
            latent_z_1 = extract_latents(len(img_test_1), model_body, path_checkpoint_1, img_test_1)

            pca = PCA(num_latent)
            pca.fit(latent_z_1)
            explained_variance_ratio.append(pca.explained_variance_ratio_)
            #plt.plot([i for i in range(1, len(pca.explained_variance_ratio_) + 1)], pca.explained_variance_ratio_,
            #         'ro-', c='red')
            print(time.perf_counter() - start_time)

        explained_variance_ratio = np.array(explained_variance_ratio)
        summary_explained_variance_ratio.append(explained_variance_ratio)

    with open(path_save_sum, 'wb') as f:
        pickle.dump(summary_explained_variance_ratio, f)

