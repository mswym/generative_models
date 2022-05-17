import numpy as np
import torch

from translucency.vae_vanilla import VAE_vanilla
from translucency.ae import AE
from translucency.class_mydataset import MyDatasetBinary
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.linear_model import LinearRegression

import time
import pickle
import matplotlib.pyplot as plt
import itertools
from PIL import Image
import os

from translucency.utils import *


def cal_linear_reg(x, new_coef_matrix, intercept):
    x_hat = np.dot(new_coef_matrix, x.T)

    return x_hat.T + intercept

def cal_decoding(model, z, num_itr):
    max_img = z.shape[0]
    step_img = round(max_img/num_itr)
    z = torch.from_numpy(z)
    out_img = []
    for itr in range(num_itr):
        ind = itr*step_img
        tmp_img = model.decoders(z[ind:ind+step_img, :])
        tmp_img = tmp_img.to('cpu').detach().numpy().copy()
        out_img.append(tmp_img)
    out_img = np.array(out_img)
    return out_img.reshape(-1, out_img.shape[2], out_img.shape[3], out_img.shape[4])

def trans_cpu_and_save(x, dir_name, mask):
    x = x.transpose(0, 2, 3, 1)
    for ind_img in range(x.shape[0]):
        tmp = 255 * np.reshape(x[ind_img], [x.shape[1], x.shape[2],x.shape[3]])
        tmp = tmp * mask
        tmp = Image.fromarray(np.uint8(tmp))

        tmp.save(dir_name + '/' + str(ind_img) + '.png', format='PNG')

def make_mask(mask):
    mask = np.array(mask)
    mask[np.where(mask<255)] = 0

    return mask/255

def batch_run_make_swapping(list_num_latent,cond='vae', cond2=''):
    batch_size_train = 30
    batch_size_test = 300
    size_input = np.array([256, 256, 3])
    scale_eigenval = 1
    flag_autoadjust_eigenval = False

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
                    'sphere']
    # list_objname = ['buddha', 'armadillo']
    # list_num_latent = [2, 4, 8, 16, 32, 64, 128, 256]
    #list_num_latent = [8]

    path_dir_save = '/media/mswym/PortableSSD/translucency/'
    model_body = VAE_vanilla()

    comb_list = list(itertools.combinations(np.linspace(0, len(list_objname) - 1, len(list_objname)), 2))
    count = 0
    for num_latent in list_num_latent:
        count = count + 1
        for ind_obj in range(len(comb_list)):

            start_time = time.perf_counter()
            ind_obj_1 = int(comb_list[ind_obj][0])
            ind_obj_2 = int(comb_list[ind_obj][1])

            path_decoder_load_1 = path_dir_save + "obj_mask/coef/coef_" + list_objname[ind_obj_1] + "-" + list_objname[
                ind_obj_2] + "_latent" + str(num_latent) + ".pkl"
            path_decoder_load_2 = path_dir_save + "obj_mask/coef/coef_" + list_objname[ind_obj_2] + "-" + list_objname[
                ind_obj_1] + "_latent" + str(num_latent) + ".pkl"

            path_checkpoint_1 = path_dir_save + "obj_mask/" + list_objname[
                ind_obj_1] + "_latent" + str(num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"
            path_checkpoint_2 = path_dir_save + "obj_mask/" + list_objname[
                ind_obj_2] + "_latent" + str(num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"



            mypath_1 = path_dir_save + 'model_objects_tonemap_mask/che_220322_300test_' + list_objname[
                ind_obj_1] + '.binary'
            mypath_2 = path_dir_save + 'model_objects_tonemap_mask/che_220322_300test_' + list_objname[
                ind_obj_2] + '.binary'

            img_transform = transforms.Compose([
                transforms.Resize((size_input[0], size_input[1])),
                transforms.ToTensor()
            ])

            img_train_1 = read_img_dataset(mypath_1, img_transform, batch_size_train)
            img_train_2 = read_img_dataset(mypath_2, img_transform, batch_size_train)

            # load models
            # confirm this can work well.
            model_1 = model_body.load_from_checkpoint(checkpoint_path=path_checkpoint_1)
            model_2 = model_body.load_from_checkpoint(checkpoint_path=path_checkpoint_2)

            # latent and non-switched generation
            # latent computing.
            latent_z_1, img_hat_1 = extract_latents_xhats(len(img_train_1), model_1, img_train_1)
            latent_z_2, img_hat_2 = extract_latents_xhats(len(img_train_2), model_2, img_train_2)

            #load decoders
            with open(path_decoder_load_1, 'rb') as f:
                decoder_12 = pickle.load(f)
            with open(path_decoder_load_2, 'rb') as f:
                decoder_21 = pickle.load(f)

            #transfer latent code
            latent_z_12 = cal_linear_reg(latent_z_1, decoder_12['coefficient'], decoder_12['intercept'])
            latent_z_21 = cal_linear_reg(latent_z_2, decoder_21['coefficient'], decoder_21['intercept'])

            #load mask image
            mask_1 = Image.open(path_dir_save + 'mask/mask_' + list_objname[ind_obj_1] + '.png')
            mask_1 = make_mask(mask_1)
            mask_2 = Image.open(path_dir_save + 'mask/mask_' + list_objname[ind_obj_2] + '.png')
            mask_2 = make_mask(mask_2)

            #make transferred image
            img_hat_12 = cal_decoding(model_2, latent_z_12, len(img_train_1))
            img_hat_21 = cal_decoding(model_1, latent_z_21, len(img_train_2))

            #save images
            dir_name = path_dir_save + "obj_mask/_results/" + list_objname[ind_obj_1] + list_objname[ind_obj_2] + '_' + str(
                scale_eigenval) + '_latent' + str(num_latent)
            #os.makedirs(dir_name + '/1', exist_ok=True)
            #os.makedirs(dir_name + '/2', exist_ok=True)
            os.makedirs(dir_name + '/11', exist_ok=True)
            os.makedirs(dir_name + '/22', exist_ok=True)
            os.makedirs(dir_name + '/12', exist_ok=True)
            os.makedirs(dir_name + '/21', exist_ok=True)
            trans_cpu_and_save(img_hat_1, dir_name + '/11', mask_1)
            trans_cpu_and_save(img_hat_2, dir_name + '/22', mask_2)
            trans_cpu_and_save(img_hat_12, dir_name + '/12', mask_2)
            trans_cpu_and_save(img_hat_21, dir_name + '/21', mask_1)
            #if count == 1:
            # #i have to transpose image_train_1 to img. it's better to make this in another file
            #    trans_cpu_and_save(img_train_1, dir_name + '/1', mask_1)
            #    trans_cpu_and_save(img_train_2,, dir_name + '/2', mask_2)

            print(time.perf_counter() - start_time)
if __name__ == '__main__':
    batch_size_train = 30
    batch_size_test = 300
    size_input = np.array([256, 256, 3])
    scale_eigenval = 1
    flag_autoadjust_eigenval = False

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
                    'sphere']
    # list_objname = ['buddha', 'armadillo']
    list_num_latent = [2, 4, 8, 16, 32, 64, 128, 256]
    #list_num_latent = [8]

    path_dir_save = '/media/mswym/PortableSSD/translucency/'
    model_body = VAE_vanilla()

    comb_list = list(itertools.combinations(np.linspace(0, len(list_objname) - 1, len(list_objname)), 2))
    count = 0
    for num_latent in list_num_latent:
        count = count + 1
        for ind_obj in range(len(comb_list)):

            start_time = time.perf_counter()
            ind_obj_1 = int(comb_list[ind_obj][0])
            ind_obj_2 = int(comb_list[ind_obj][1])

            path_decoder_load_1 = path_dir_save + "obj_mask/coef/coef_" + list_objname[ind_obj_1] + "-" + list_objname[
                ind_obj_2] + "_latent" + str(num_latent) + ".pkl"
            path_decoder_load_2 = path_dir_save + "obj_mask/coef/coef_" + list_objname[ind_obj_2] + "-" + list_objname[
                ind_obj_1] + "_latent" + str(num_latent) + ".pkl"

            path_checkpoint_1 = path_dir_save + "obj_mask/" + list_objname[
                ind_obj_1] + "_latent" + str(num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"
            path_checkpoint_2 = path_dir_save + "obj_mask/" + list_objname[
                ind_obj_2] + "_latent" + str(num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"



            mypath_1 = path_dir_save + 'model_objects_tonemap_mask/che_220322_300test_' + list_objname[
                ind_obj_1] + '.binary'
            mypath_2 = path_dir_save + 'model_objects_tonemap_mask/che_220322_300test_' + list_objname[
                ind_obj_2] + '.binary'

            img_transform = transforms.Compose([
                transforms.Resize((size_input[0], size_input[1])),
                transforms.ToTensor()
            ])

            img_train_1 = read_img_dataset(mypath_1, img_transform, batch_size_train)
            img_train_2 = read_img_dataset(mypath_2, img_transform, batch_size_train)

            # load models
            # confirm this can work well.
            model_1 = model_body.load_from_checkpoint(checkpoint_path=path_checkpoint_1)
            model_2 = model_body.load_from_checkpoint(checkpoint_path=path_checkpoint_2)

            # latent and non-switched generation
            # latent computing.
            latent_z_1, img_hat_1 = extract_latents_xhats(len(img_train_1), model_1, img_train_1)
            latent_z_2, img_hat_2 = extract_latents_xhats(len(img_train_2), model_2, img_train_2)

            #load decoders
            with open(path_decoder_load_1, 'rb') as f:
                decoder_12 = pickle.load(f)
            with open(path_decoder_load_2, 'rb') as f:
                decoder_21 = pickle.load(f)

            #transfer latent code
            latent_z_12 = cal_linear_reg(latent_z_1, decoder_12['coefficient'], decoder_12['intercept'])
            latent_z_21 = cal_linear_reg(latent_z_2, decoder_21['coefficient'], decoder_21['intercept'])

            #load mask image
            mask_1 = Image.open(path_dir_save + 'mask/mask_' + list_objname[ind_obj_1] + '.png')
            mask_1 = make_mask(mask_1)
            mask_2 = Image.open(path_dir_save + 'mask/mask_' + list_objname[ind_obj_2] + '.png')
            mask_2 = make_mask(mask_2)

            #make transferred image
            img_hat_12 = cal_decoding(model_2, latent_z_12, len(img_train_1))
            img_hat_21 = cal_decoding(model_1, latent_z_21, len(img_train_2))

            #save images
            dir_name = path_dir_save + "obj_mask/_results/" + list_objname[ind_obj_1] + list_objname[ind_obj_2] + '_' + str(
                scale_eigenval) + '_latent' + str(num_latent)
            #os.makedirs(dir_name + '/1', exist_ok=True)
            #os.makedirs(dir_name + '/2', exist_ok=True)
            os.makedirs(dir_name + '/11', exist_ok=True)
            os.makedirs(dir_name + '/22', exist_ok=True)
            os.makedirs(dir_name + '/12', exist_ok=True)
            os.makedirs(dir_name + '/21', exist_ok=True)
            trans_cpu_and_save(img_hat_1, dir_name + '/11', mask_1)
            trans_cpu_and_save(img_hat_2, dir_name + '/22', mask_2)
            trans_cpu_and_save(img_hat_12, dir_name + '/12', mask_2)
            trans_cpu_and_save(img_hat_21, dir_name + '/21', mask_1)
            #if count == 1:
            # #i have to transpose image_train_1 to img. it's better to make this in another file
            #    trans_cpu_and_save(img_train_1, dir_name + '/1', mask_1)
            #    trans_cpu_and_save(img_train_2,, dir_name + '/2', mask_2)

            print(time.perf_counter() - start_time)