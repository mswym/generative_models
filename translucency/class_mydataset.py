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

import re


# these classes consider reading the images rendered by Che et al.(2020)

class MyDatasetDir(torch.utils.data.Dataset):
    def __init__(self, mypath, path_mask=0, val_mask=1, transform1=None, flag_hdr=False,
                 fname_save='my_dataset.binaryfile'):
        self.transform1 = transform1
        self.flag_hdr = flag_hdr
        class_imgs = ReadImgDir(mypath, path_mask, val_mask, flag_hdr=True)
        self.dataset, self.label = class_imgs.read_all_img()

        with open(fname_save, 'wb') as f:
            pickle.dump(self.dataset, f)
        tmp_dir, tmp_file = os.path.split(fname_save)
        with open(tmp_dir + '/label_' + tmp_file, 'wb') as f:
            pickle.dump(self.label, f)
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

        # return out_data, out_label
        return out_data, out_label


class ReadImgDir():
    def __init__(self, path_dir, path_mask, val_mask, flag_hdr=False, val_gamma = 2.2):
        self.fname_list_img = glob.glob(path_dir)
        self.fname_list_img.sort()
        self.flag_hdr = flag_hdr
        self.tonemap = cv.createTonemap(gamma=val_gamma)
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
        if self.flag_hdr:
            for i_list in range(self.num_img):
                print(self.fname_list_img[i_list])
                img = self.read_exr_fnc(self.fname_list_img[i_list])  # size is m x n x c
                img = self.apply_tonemap(img)
                img = img * self.maskimg
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
        return list_img, list_label

    def apply_tonemap(self,img):
        return self.tonemap.process(img.copy())
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

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth']
    # list_objname = ['d36','d44','d54','d67','d82','d100','d122','d150','d184','d225','d276']
    # list_objname = ['a0.39','a0.59','a0.74','a0.87','a0.95']
    # list_objname = ['d276']
    list_mask_val = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    path_dir = '/media/mswym/SSD-PGU3/database/translucent_data_che/'
    path_dir_model = '/media/mswym/SSD-PGU3/database/results_translucent_2022/model_objects_notnormalize/'

    for ind_obj in list_objname):
        path_img = path_dir + 'split_objects/' + ind_obj + '/*.exr'
        # path_img = path_dir + 'split_objects/test_' + list_objname[ind_obj] + '/*.exr'
        path_mask = path_dir + 'mask/' + ind_obj + '.exr'

        fname_save_binary = path_dir_model + 'che_03112021_1500train_' + ind_obj + '.binary'
        # fname_save_binary = path_dir_model+'che_03112021_300test_'+ list_objname[ind_obj] + '.binary'

        val_dataset = MyDatasetDir(path_img, path_mask=path_mask, val_mask=ind_obj, transform1=None,
                                        flag_hdr=True, fname_save=fname_save_binary)
