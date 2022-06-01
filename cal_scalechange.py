import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import torch

from translucency.utils import *
from translucency.class_mydataset import MyDatasetBinary
from translucency.vae_vanilla import VAE_vanilla

from sklearn import svm
from PIL import Image

import itertools
import seaborn as sns

def extract_opaque_trans(dataloader):
    imgs = []
    labels = []
    for data in dataloader:
        img, label = data
        if label[0][2] == 276:  # opaque
            imgs.append(img)
            labels.append(0)
        elif label[0][2] == 36:  # translucent
            imgs.append(img)
            labels.append(1)


    return imgs, labels

def extract_latent_xhat(model, img_train):
    latent_z, img_hat, _, _ = model._run_step(img_train)

    return latent_z.to('cpu').detach().numpy().copy(), img_hat.to('cpu').detach().numpy().copy()

def extract_latents_xhats_imgs(num_itr, model, img_train):
    model.eval()
    latent_z = []
    x_hat = []
    for img in img_train:
        tmp_z, tmp_xhat = extract_latent_xhat(model, img)
        latent_z.append(tmp_z)
        x_hat.append(tmp_xhat)

    latent_z = np.array(latent_z)
    x_hat = np.array(x_hat)
    return latent_z.reshape([-1, latent_z.shape[2]]), x_hat.reshape(
        [-1, x_hat.shape[2], x_hat.shape[3], x_hat.shape[4]])

def cal_linear_reg(x, new_coef_matrix, intercept):
    x_hat = np.dot(new_coef_matrix, x.T)

    return x_hat.T + intercept


if __name__ == '__main__':
    size_input = np.array([256, 256, 3])
    scale_eigenval = 1
    flag_orig_save = True
    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
                    'sphere']

    path_dir_save = '/home/mswym/workspace/db/'

    model_body = VAE_vanilla()
    list_param_scale = [1]
    list_num_latent = [32]
    comb_list = list(itertools.combinations(np.linspace(0, len(list_objname) - 1, len(list_objname)), 2))

    for num_latent in list_num_latent:
        for param_scale in list_param_scale:
            mat_scale = np.ones((len(list_objname),len(list_objname)))
            for ind_obj in range(len(comb_list)):
                start_time = time.perf_counter()
                ind_obj_1 = list_objname[int(comb_list[ind_obj][0])]
                ind_obj_2 = list_objname[int(comb_list[ind_obj][1])]
                print(ind_obj_1)
                print(ind_obj_2)

                path_load_1 = path_dir_save + "obj_mask/svm/svm_" + ind_obj_1 + "_latent" + str(
                    num_latent) + ".pkl"
                path_load_2 = path_dir_save + "obj_mask/svm/svm_" + ind_obj_2 + "_latent" + str(
                    num_latent) + ".pkl"



                # load trained svm models
                with open(path_load_1, 'rb') as f:
                    svm_model_1 = pickle.load(f)
                with open(path_load_2, 'rb') as f:
                    svm_model_2 = pickle.load(f)

                ratio_12 = np.linalg.norm(svm_model_1.coef_)/np.linalg.norm(svm_model_2.coef_)
                ratio_21 = np.linalg.norm(svm_model_2.coef_) / np.linalg.norm(svm_model_1.coef_)

                mat_scale[int(comb_list[ind_obj][0]), int(comb_list[ind_obj][1])] = copy.deepcopy(ratio_12)
                mat_scale[int(comb_list[ind_obj][1]), int(comb_list[ind_obj][0])] = copy.deepcopy(ratio_21)

            with open(path_dir_save + 'mat_scale' + str(num_latent) + '.pkl', 'wb') as f:
                pickle.dump(mat_scale, f)

            axes = sns.heatmap(np.log(mat_scale), cmap="PuOr_r", center=0)
            plt.show()
