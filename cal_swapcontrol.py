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

def cal_decoding(model, z, num_itr):
    max_img = z.shape[0]
    step_img = round(max_img/num_itr)
    z = torch.from_numpy(z)
    z = z.to(torch.float)
    out_img = []
    for itr in range(num_itr):
        ind = itr*step_img
        tmp_img = model.decoders(z[ind:ind+step_img, :])
        tmp_img = tmp_img.to('cpu').detach().numpy().copy()
        out_img.append(tmp_img)
    out_img = np.array(out_img)
    return out_img.reshape(-1, out_img.shape[2], out_img.shape[3], out_img.shape[4])

def make_mask(mask):
    mask = np.array(mask)
    mask[np.where(mask<255)] = 0

    return mask/255

def trans_cpu_and_save(x, dir_name, mask):
    x = x.transpose(0, 2, 3, 1)
    for ind_img in range(x.shape[0]):
        tmp = 255 * np.reshape(x[ind_img], [x.shape[1], x.shape[2],x.shape[3]])
        tmp = tmp * mask
        tmp = Image.fromarray(np.uint8(tmp))

        tmp.save(dir_name + '/' + str(ind_img) + '.png', format='PNG')

def oritinal_trans_cpu_and_save(num_itr, img_train, dir_name):
    for i,img in enumerate(img_train):
        img = img[0].to('cpu').detach().numpy().copy()
        img = img.transpose(1,2,0)
        img = 255 * img
        img = Image.fromarray(np.uint8(img))

        img.save(dir_name + '/' + str(i) + '.png', format='PNG')

if __name__ == '__main__':
    size_input = np.array([256, 256, 3])
    scale_eigenval = 1
    flag_autoadjust_eigenval = False
    flag_orig_save = True
    # list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
    #                'sphere']
    # list_num_latent = [16]
    list_objname1 = ['lucy', 'lucy', 'bust', 'bust', 'buddha', 'buddha']
    list_objname2 = ['bun', 'sphere', 'bun', 'sphere', 'bun', 'sphere']

    path_dir_save = '/home/mswym/workspace/db/'

    model_body = VAE_vanilla()
    #ind_obj_1 = 'lucy'
    #ind_obj_2 = 'bun'
    #param_scale = 1
    list_param_scale = [9,10,11,12]
    list_num_latent = [32]

    for num_latent in list_num_latent:
        for param_scale in list_param_scale:
            for ind_obj_1,ind_obj_2 in zip(list_objname1,list_objname2):
                print(ind_obj_1)
                print(ind_obj_2)

                start_time = time.perf_counter()

                path_load_1 = path_dir_save + "obj_mask/svm/svm_" + ind_obj_1 + "_latent" + str(
                    num_latent) + ".pkl"
                path_load_2 = path_dir_save + "obj_mask/svm/svm_" + ind_obj_2 + "_latent" + str(
                    num_latent) + ".pkl"

                path_checkpoint_1 = path_dir_save + "obj_mask/" + ind_obj_1 + "_latent" + str(
                    num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"
                path_checkpoint_2 = path_dir_save + "obj_mask/" + ind_obj_2 + "_latent" + str(
                    num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"

                mypath_1 = path_dir_save + 'model_objects_tonemap_mask/che_220322_300test_' + ind_obj_1 + '.binary'
                mypath_2 = path_dir_save + 'model_objects_tonemap_mask/che_220322_300test_' + ind_obj_2 + '.binary'

                path_decoder_load_1 = path_dir_save + "obj_mask/coef/coef_" + ind_obj_1 + "-" + ind_obj_2 + "_latent" + str(num_latent) + ".pkl"
                path_decoder_load_2 = path_dir_save + "obj_mask/coef/coef_" + ind_obj_2 + "-" + ind_obj_1 + "_latent" + str(num_latent) + ".pkl"


                img_transform = transforms.Compose([
                    transforms.Resize((size_input[0], size_input[1])),
                    transforms.ToTensor()
                ])

                # load dataset
                img_test_1 = read_img_dataset(mypath_1, img_transform, 1)  # batche size 1
                img_test_2 = read_img_dataset(mypath_2, img_transform, 1)  # batche size 1

                # extract only opaque and translucenct images
                img_test_1, labels_1 = extract_opaque_trans(img_test_1)
                img_test_2, labels_2 = extract_opaque_trans(img_test_2)

                # calculate z value
                model_1 = model_body.load_from_checkpoint(checkpoint_path=path_checkpoint_1)
                model_2 = model_body.load_from_checkpoint(checkpoint_path=path_checkpoint_2)
                latent_z_1, img_hat_1 = extract_latents_xhats_imgs(len(img_test_1), model_1, img_test_1)
                latent_z_2, img_hat_2 = extract_latents_xhats_imgs(len(img_test_2), model_2, img_test_2)

                # load trained svm models
                with open(path_load_1, 'rb') as f:
                    svm_model_1 = pickle.load(f)
                with open(path_load_2, 'rb') as f:
                    svm_model_2 = pickle.load(f)

                #dist_1 = svm_model_1.decision_function(latent_z_1)/np.linalg.norm(svm_model_1.coef_)
                #dist_2 = svm_model_2.decision_function(latent_z_2) / np.linalg.norm(svm_model_2.coef_)

                unit_1 = svm_model_1.coef_/np.linalg.norm(svm_model_1.coef_)
                unit_2 = svm_model_2.coef_ / np.linalg.norm(svm_model_2.coef_)
                latent_z_1 = latent_z_1 + param_scale * unit_1
                latent_z_2 = latent_z_2 + param_scale * unit_2

                #swap z
                # load decoders
                with open(path_decoder_load_1, 'rb') as f:
                    decoder_12 = pickle.load(f)
                with open(path_decoder_load_2, 'rb') as f:
                    decoder_21 = pickle.load(f)

                # transfer latent code
                latent_z_12 = cal_linear_reg(latent_z_1, decoder_12['coefficient'], decoder_12['intercept'])
                latent_z_21 = cal_linear_reg(latent_z_2, decoder_21['coefficient'], decoder_21['intercept'])

                # load mask image
                mask_1 = Image.open(path_dir_save + 'mask/mask_' + ind_obj_1 + '.png')
                mask_1 = make_mask(mask_1)
                mask_2 = Image.open(path_dir_save + 'mask/mask_' + ind_obj_2 + '.png')
                mask_2 = make_mask(mask_2)

                #generate imgs using generator
                # make transferred image
                img_hat_12 = cal_decoding(model_2, latent_z_12, len(img_test_1))
                img_hat_21 = cal_decoding(model_1, latent_z_21, len(img_test_2))
                img_hat_1 = cal_decoding(model_1, latent_z_1, len(img_test_1))
                img_hat_2 = cal_decoding(model_2, latent_z_2, len(img_test_2))

                # save images
                dir_name = path_dir_save + "obj_mask/_results_scale/_results_scale" + str(param_scale) + "/" + ind_obj_1 + ind_obj_2 + '_' + str(
                    scale_eigenval) + '_latent' + str(num_latent)

                os.makedirs(dir_name + '/11', exist_ok=True)
                os.makedirs(dir_name + '/22', exist_ok=True)
                os.makedirs(dir_name + '/12', exist_ok=True)
                os.makedirs(dir_name + '/21', exist_ok=True)
                trans_cpu_and_save(img_hat_1, dir_name + '/11', mask_1)
                trans_cpu_and_save(img_hat_2, dir_name + '/22', mask_2)
                trans_cpu_and_save(img_hat_12, dir_name + '/12', mask_2)
                trans_cpu_and_save(img_hat_21, dir_name + '/21', mask_1)

                if flag_orig_save:
                    os.makedirs(dir_name + '/1', exist_ok=True)
                    os.makedirs(dir_name + '/2', exist_ok=True)
                    oritinal_trans_cpu_and_save(len(img_test_1), img_test_1, dir_name + '/1')
                    oritinal_trans_cpu_and_save(len(img_test_2), img_test_2, dir_name + '/2')

                print(time.perf_counter() - start_time)
            # control the boundary values
