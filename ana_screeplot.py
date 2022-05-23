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
import matplotlib.ticker as ticker

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


def batch_ana_screeplot():
    batch_size_train = 10
    size_input = np.array([256, 256, 3])
    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
                    'sphere']

    path_dir_save = '/media/mswym/PortableSSD/translucency/'
    path_save_sum = path_dir_save + 'vae_vanilla_screeplot.pickle'
    model_body = VAE_vanilla()
    list_num_latent = [2, 4, 8, 16, 32, 64, 128, 256]
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

def draw_plots(summary_explained_variance_ratio, std_val=0.05, list_xlim=[0.7, 64],
                           list_ylim=[0, 0.9],
                           list_set_xticklabels=['', '', '1', '10', '100'], list_set_xticks=[1, 10, 100],
                           list_set_yticklabels=['0.0', '0.2', '0.4', '0.6', '0.8'],
                           list_set_yticks=[0, 0.2, 0.4, 0.6, 0.8],
                           val_ticks=0.02,
                           fname_save='screeplot.png',
                           scale_jitter=5,
                           scale_panel=1,
                           alpha_fill=0.2):


    fig = plt.figure(figsize=(8.7, 6.5))
    axes = fig.add_subplot(1, scale_panel, 1)
    num_params = len(summary_explained_variance_ratio)
    col_map = plt.get_cmap('plasma')
    for i in range(num_params):
        data = summary_explained_variance_ratio[i]
        mean = np.mean(data, 0)
        sem = np.std(data, 0)/np.sqrt(data.shape[0])
        x = np.linspace(1,len(mean),len(mean))
        ind_color = int(i*(255/num_params))
        print(ind_color)
        print(col_map(ind_color))
        axes.plot(x, mean, color=col_map(ind_color), linewidth=3)
        axes.fill_between(x, mean+sem, mean-sem, color=col_map(ind_color), alpha=alpha_fill)
        #axes.errorbar(x, mean,
        #              yerr=[std, std],
        #              capsize=15,
        #              fmt='s', markersize=11, ecolor='red', markeredgecolor="black", color='red', mew=1)


    axes.set_xlim(list_xlim)
    axes.set_ylim(list_ylim)
    # axes.xaxis.grid(True, which='minor')
    #axes.set_xticks(list_set_xticks)
    axes.set_xscale('log')
    axes.set_xticklabels(list_set_xticklabels, fontsize=20)
    axes.set_yticks(list_set_yticks)
    axes.set_yticklabels(list_set_yticklabels, fontsize=20)
    # axes.grid()
    axes.yaxis.set_minor_locator(ticker.MultipleLocator(val_ticks))
    fig.savefig(fname_save)
    plt.show()

if __name__ == '__main__':
    path_dir_save = '/media/mswym/PortableSSD/translucency/'
    path_save_sum = path_dir_save + 'vae_vanilla_screeplot.pickle'

    with open(path_save_sum, 'rb') as f:
        summary_explained_variance_ratio = pickle.load(f)

    draw_plots(summary_explained_variance_ratio)

