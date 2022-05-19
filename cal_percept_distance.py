import glob
from PIL import Image, ImageChops, ImageStat
import numpy as np

# image metrics
# LPIPS
# https://github.com/richzhang/PerceptualSimilarity

# DISTS
# https://github.com/dingkeyan93/DISTS

# PSNR

import lpips
import torch
from torchvision import transforms

# from DISTS_pytorch import DISTS
import itertools

# visualize
import seaborn as sns
import matplotlib.pyplot as plt


def psnr_imgs(img1, img2):
    # rgb images
    diff_img = ImageChops.difference(img1, img2)
    stat = ImageStat.Stat(diff_img)
    mse = sum(stat.sum2) / len(stat.count) / sum(stat.count)
    return np.array(10 * np.log10(255 ** 2 / mse))


def cal_lpips(img1, img2, loss_fn_alex):
    # rgb images
    # loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

    # img1 = img_transform(img1)
    # img2 = img_transform(img2)
    distance = loss_fn_alex(img1, img2)
    distance = distance.to('cpu').detach().numpy().copy()

    return distance


def min_max_normalization(tensor, min_value, max_value):
    # to 0-1
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def make_rdm(vals, comb_list, length_mat, same_val=0):
    mat_rdm = same_val * np.ones((length_mat, length_mat))
    for i, ind_obj in enumerate(comb_list):
        mat_rdm[int(ind_obj[0]), int(ind_obj[1])] = vals[i]
        mat_rdm[int(ind_obj[1]), int(ind_obj[0])] = vals[i]

    return mat_rdm


def draw_rdm(vals, name_rowcol):
    mask = np.zeros_like(vals)
    mask[np.triu_indices_from(mask)]

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask,
                         vmin=-1.0,
                         vmax=1.0,
                         cmap='plasma',
                         annot=False,
                         fmt='.1f',  # when I put annot
                         xticklabels=name_rowcol,
                         yticklabels=name_rowcol,
                         square=True)

    plt.show()


if __name__ == '__main__':
    num_img = 300
    size_img = [256, 256, 3]
    path_dir = '/home/mswym/workspace/db/ae_fail/'

    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores

    trans_for_lpips = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda tensor: min_max_normalization(tensor, -1, 1)),
    ])

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
                    'sphere']
    comb_list = list(itertools.combinations(np.linspace(0, len(list_objname) - 1, len(list_objname)), 2))


    #list_condcomp = [(1, 11), (1, 21), (2, 12), (2, 22)]
    list_condcomp = [(1, 11), (1, 21)]

    list_num_latent = [2, 4, 16, 64, 128, 256]

    for num_latent in list_num_latent:
        for i, ind_cond in enumerate(list_condcomp):
            means_psnr = []
            means_lpips = []
            for ind_obj in comb_list:
                psnr = []
                lpips = []
                #if i == 0 or i == 1:
                if i == 99:
                    path_dir_1 = path_dir + 'obj_mask/_results/' + list_objname[int(ind_obj[0])] + \
                                 list_objname[int(ind_obj[1])] + '_1' + '_latent' + str(num_latent) + '/' + str(ind_cond[0]) + '/*'
                    path_dir_2 = path_dir + 'obj_mask/_results/' + list_objname[int(ind_obj[0])] + \
                                 list_objname[int(ind_obj[1])] + '_1' + '_latent' + str(num_latent) + '/' + str(ind_cond[1]) + '/*'
                else:
                    path_dir_1 = path_dir + 'imgs_test/' + list_objname[int(ind_obj[int(ind_cond[0]-1)])] + '/*'
                    path_dir_2 = path_dir + 'obj_mask/_results/' + list_objname[int(ind_obj[0])] + list_objname[
                        int(ind_obj[1])] + '_1' + '_latent' + str(num_latent) + '/' + str(ind_cond[1]) + '/*'

                fname_list_img_1 = glob.glob(path_dir_1)
                fname_list_img_1.sort()

                fname_list_img_2 = glob.glob(path_dir_2)
                fname_list_img_2.sort()

                print(path_dir_1)
                print(path_dir_2)
                for fname_img_1, fname_img_2 in zip(fname_list_img_1, fname_list_img_2):
                    img_1 = Image.open(fname_img_1)
                    img_2 = Image.open(fname_img_2)

                    img_1 = img_1.convert('RGB')
                    img_2 = img_2.convert('RGB')

                    # psnr
                    psnr.append(psnr_imgs(img_1, img_2))
                    # lpips
                    lpips.append(cal_lpips(trans_for_lpips(img_1), trans_for_lpips(img_2), loss_fn_alex))

                fname_tmp = path_dir + 'obj_mask/_results/' + list_objname[int(ind_obj[0])] + list_objname[
                        int(ind_obj[1])] + '_1' + '_latent' + str(num_latent) + '/_individual_psnr_lpips.npz'
                np.savez(fname_tmp, psnr, lpips) #this is wrong

                means_psnr.append(np.mean(psnr))
                means_lpips.append(np.mean(lpips))

            rdm_psnr = make_rdm(means_psnr, comb_list, len(list_objname), same_val=100)
            rdm_lpips = make_rdm(means_lpips, comb_list, len(list_objname), same_val=0)
            fname_save = path_dir + 'rdm_latent' + str(num_latent) + '_stim1_' + str(ind_cond[0]) + '_stim2_' + str(ind_cond[1]) + '.npz'
            np.savez(fname_save, rdm_psnr, rdm_lpips)
