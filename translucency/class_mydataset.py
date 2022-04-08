import os
import torch
import glob
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt

# for openexr reading
import array
import OpenEXR
import Imath

import cv2
import re


# these classes consider reading the images rendered by Che et al.(2020)

class MyDatasetDir(torch.utils.data.Dataset):
    def __init__(self, mypath, path_mask=0, val_mask=1, transform1=None, flag_hdr=False,
                 fname_save='my_dataset.binaryfile'):
        self.transform1 = transform1
        self.flag_hdr = flag_hdr
        class_imgs = ReadImgDir(mypath, path_mask, val_mask, flag_hdr=True)
        self.dataset, self.label, self.mean_img, self.std_img = class_imgs.read_all_img()

        with open(fname_save, 'wb') as f:
            pickle.dump(self.dataset, f)
        tmp_dir, tmp_file = os.path.split(fname_save)
        with open(tmp_dir + '/label_' + tmp_file, 'wb') as f:
            pickle.dump(self.label, f)
        with open(tmp_dir + '/mean_' + tmp_file, 'wb') as f:
            pickle.dump(self.mean_img, f)
        with open(tmp_dir + '/std_' + tmp_file, 'wb') as f:
            pickle.dump(self.std_img, f)
        self.datanum = len(self.dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.dataset[idx]
        # out_data = torch.from_numpy(out_data).float()
        out_label = self.label[idx]
        # out_label = torch.from_numpy(out_label).long()

        if self.transform1:
            out_data = self.transform1(out_data)

        return out_data, out_label


class MyDatasetBinary(torch.utils.data.Dataset):
    def __init__(self, mypath, transform1=None, flag_hdr=False):
        self.transform1 = transform1
        self.flag_hdr = flag_hdr

        with open(mypath, 'rb') as f:
            self.dataset = pickle.load(f)
        tmp_dir, tmp_file = os.path.split(mypath)
        with open(tmp_dir + '/label_' + tmp_file, 'rb') as f:
            self.label = pickle.load(f)
        with open(tmp_dir + '/mean_' + tmp_file, 'rb') as f:
            self.mean_img = pickle.load(f)
        with open(tmp_dir + '/std_' + tmp_file, 'rb') as f:
            self.std_img = pickle.load(f)

        self.datanum = len(self.dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.dataset[idx].convert('RGB')
        # out_data = self.dataset[idx]
        out_label = self.label[idx]

        if self.transform1:
            out_data = self.transform1(out_data)

        # return out_data, out_label
        return out_data, out_label


class ReadImgDir():
    def __init__(self, path_dir, path_mask, val_mask, flag_hdr=False, val_gamma=2.2):
        self.fname_list_img = glob.glob(path_dir)
        self.fname_list_img.sort()
        self.flag_hdr = flag_hdr
        self.num_img = len(self.fname_list_img)
        if path_mask != 0:
            self.maskimg = self.read_exr_fnc(path_mask)
            if val_mask == 0:
                self.maskimg = np.abs(self.maskimg - 1)
        else:
            self.maskimg = 1
        print('num of images is ' + str(self.num_img))

    def read_all_img(self):
        # print(list_img)
        list_img = []
        list_label = []
        mean_img = []
        std_img = []
        if self.flag_hdr:
            for i_list in range(self.num_img):
                print(self.fname_list_img[i_list])
                img = self.read_exr_fnc(self.fname_list_img[i_list])  # size is m x n  here only grayscale
                img = self.apply_tonemap_exposure(img)
                img = 255 * img * self.maskimg
                if len(img.shape):
                    mean_img.append(np.mean(img))
                    std_img.append(np.std(img))
                else:
                    mean_img.append(np.mean(img, 2))
                    std_img.append(np.std(img, 2))

                img = Image.fromarray(img.astype('uint8'))

                tmp_label1 = re.findall(r'\d+', self.fname_list_img[i_list])
                tmp_label2 = re.findall(r'\d+\.\d+', self.fname_list_img[i_list])

                labels = np.zeros((5,))
                labels[0] = i_list
                labels[1] = tmp_label1[1]
                labels[2] = tmp_label1[2]
                labels[3] = tmp_label2[0]
                labels[4] = tmp_label2[1]

                list_img.append(img)
                list_label.append(labels)
        else:
            for i_list in range(self.num_img):
                print(self.fname_list_img[i_list])
                img = Image.open(self.fname_list_img[i_list])
                list_img.append(img)
                list_label.append(self.fname_list_img[i_list])

        mean_img = np.mean(mean_img)
        std_img = np.mean(std_img)

        return list_img, list_label, mean_img, std_img

    def apply_tonemap_exposure(self, img, param_expo=6, mean_scene=0.036, gamma=2.2):
        # a simple exposure and gamma tonemapping.
        # precomputed the mean of background.
        key = param_expo * mean_scene
        img = img / key
        img[np.where(img > 1)] = 1
        return img ** (1 / gamma)

    def apply_tonemap_global(self, img, eps=10 ** (-10), param_a=1):
        # a global tone mapping using geometric mean.
        key = np.exp(np.mean(np.log(eps + img)))
        img = (param_a / key) * img
        return img / (1 + img)

    def read_exr_fnc(self, fname_img):
        # to read grayscale exr images

        file_exr = OpenEXR.InputFile(fname_img)
        # Compute the size
        dw = file_exr.header()['dataWindow']
        sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # Read the three color channels as 32-bit floats
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        # (R,G,B) = [array.array('f', file_exr.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]
        (Y) = array.array('f', file_exr.channel('Y', FLOAT)).tolist()

        # R = np.reshape(np.array(R),sz)
        # G = np.reshape(np.array(G),sz)
        # B = np.reshape(np.array(B),sz)
        img = np.reshape(np.array(Y), sz)
        # img = cv2.merge((R,G,B))

        return img


if __name__ == '__main__':
    # I assume that the dataset images are stored in a directory, already separated in train and test dataset.

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    # list_objname = ['sphere']
    list_mask_val = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    # list_mask_val = [1]
    path_dir = '/media/mswym/SSD-PGU3/database/translucent_data_che/'
    path_dir_model = '/media/mswym/SSD-PGU3/database/results_translucent_220303/model_objects_tonemap_mask/'

    for i, ind_obj in enumerate(list_objname):
        # packing the training dataset from the directory
        path_img = path_dir + 'split_objects/' + ind_obj + '/*.exr'
        path_mask = path_dir + 'mask/' + ind_obj + '.exr'
        fname_save_binary = path_dir_model + 'che_220322_1500train_' + ind_obj + '.binary'

        val_dataset_train = MyDatasetDir(path_img, path_mask=path_mask, val_mask=list_mask_val[i], transform1=None,
                                         flag_hdr=True, fname_save=fname_save_binary)

        # packing the test dataset from the directory
        path_img = path_dir + 'split_objects/test_' + ind_obj + '/*.exr'
        fname_save_binary = path_dir_model + 'che_220322_300test_' + ind_obj + '.binary'

        val_dataset_test = MyDatasetDir(path_img, path_mask=path_mask, val_mask=list_mask_val[i], transform1=None,
                                        flag_hdr=True, fname_save=fname_save_binary)
