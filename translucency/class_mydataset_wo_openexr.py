import os
import torch
import glob
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt

import cv2
import re


# these classes consider reading the images rendered by Che et al.(2020)

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


if __name__ == '__main__':
    # I assume that the dataset images are stored in a directory, already separated in train and test dataset.

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    # list_objname = ['sphere']
    list_mask_val = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    # list_mask_val = [1]
    path_dir = '/media/mswym/SSD-PGU3/database/translucent_data_che/'
    path_dir_model = '/home/mswym/workspace/db/model_objects_tonemap_mask_crop/'
    flag_crop = True
    list_size_crop = [64, 128, 256] # -> applied to 512 x 512 images. Usually, the training is downsampled to 256 x 256

    for size_crop in list_size_crop:
        for i, ind_obj in enumerate(list_objname):
            # packing the training dataset from the directory
            path_img = path_dir + 'split_objects/' + ind_obj + '/*.exr'
            path_mask = path_dir + 'mask/' + ind_obj + '.exr'
            if flag_crop==True:
                fname_save_binary = path_dir_model + 'che_220322_1500train_' + ind_obj + '_cropsize_' + str(size_crop) + '.binary'
            else:
                fname_save_binary = path_dir_model + 'che_220322_1500train_' + ind_obj + '.binary'

            val_dataset_train = MyDatasetDir(path_img, path_mask=path_mask, val_mask=list_mask_val[i], transform1=None,
                                             flag_hdr=True, fname_save=fname_save_binary, flag_crop=flag_crop, size_crop=size_crop)

            # packing the test dataset from the directory
            path_img = path_dir + 'split_objects/test_' + ind_obj + '/*.exr'
            if flag_crop==True:
                fname_save_binary = path_dir_model + 'che_220322_300test_' + ind_obj + '_cropsize_' + str(size_crop) + '.binary'
            else:
                fname_save_binary = path_dir_model + 'che_220322_300test_' + ind_obj + '.binary'

            val_dataset_test = MyDatasetDir(path_img, path_mask=path_mask, val_mask=list_mask_val[i], transform1=None,
                                            flag_hdr=True, fname_save=fname_save_binary, flag_crop=flag_crop, size_crop=size_crop)
