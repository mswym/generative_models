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



def read_exr_fnc(fname_img):
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
    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    list_Val_mask = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    path_dir = '/media/mswym/SSD-PGU3/database/translucent_data_che/'
    path_save = '/media/mswym/PortableSSD/translucency/mask'

    for i, ind_obj in enumerate(list_objname):
        path_mask = path_dir + 'mask/' + ind_obj + '.exr'
        maskimg = read_exr_fnc(path_mask)
        if list_Val_mask[i] == 0:
            maskimg = np.abs(maskimg - 1)
        maskimg[np.where(maskimg>0)] = 255
        maskimg = cv2.merge((maskimg,maskimg,maskimg))
        maskimg = Image.fromarray(np.uint8(maskimg))
        maskimg = maskimg.resize((256,256))
        maskimg.save(path_save + '/mask_' + ind_obj + '.png', format='PNG')
