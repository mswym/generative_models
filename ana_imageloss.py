import glob
from PIL import Image, ImageChops, ImageStat
import numpy as np

# image metrics
# LPIPS
# https://github.com/richzhang/PerceptualSimilarity

# DISTS
# https://github.com/dingkeyan93/DISTS

import itertools

# visualize
import seaborn as sns
import matplotlib.pyplot as plt

import copy
from matplotlib.ticker import MultipleLocator


def make_rdm(vals, comb_list, length_mat, same_val = 0):
    mat_rdm = same_val*np.ones((length_mat,length_mat))
    for i, ind_obj in enumerate(comb_list):
        mat_rdm[int(ind_obj[0]), int(ind_obj[1])] = vals[i]
        mat_rdm[int(ind_obj[1]), int(ind_obj[0])] = vals[i]

    return mat_rdm

def draw_rdm(vals,name_rowcol):
    mask = np.zeros_like(vals)
    mask[np.triu_indices_from(mask)]

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask,
                         vmin=-1.0,
                         vmax=1.0,
                         cmap='plasma',
                         annot=False,
                         fmt='.1f', #when I put annot
                         xticklabels=name_rowcol,
                         yticklabels=name_rowcol,
                         square=True)

    plt.show()


def draw_distributions(mean_y1, std_y1, mean_y2, std_y2, list_xlim=[1, 384],
                           list_ylim=[0, 0.15],
                           list_set_xticklabels=['2', '4', '8', '16', '32', '64', '128', '256'],
                           list_set_xticks=[2, 4, 8, 16, 32, 64, 128, 256],
                           list_set_yticklabels=['0.00', '0.05', '0.10', '0.15'],
                           list_set_yticks=[0, 0.05, 0.1, 0.15],
                           val_ticks=0.02,
                           fname_save='workingmemory_accuracy.png',
                           scale_jitter=5,
                           scale_panel=1):

    x_sum = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    fig = plt.figure(figsize=(6.5, 6.5))
    axes = fig.add_subplot(1, scale_panel, 1)

    axes.errorbar(x_sum, mean_y1,
                  yerr=[std_y1, std_y1], capsize=15,
                  fmt='s', markersize=13, ecolor='red', markeredgecolor="black", color='red', mew=1)

    axes.plot(x_sum, mean_y1, c='red', linewidth=0.5)

    axes.errorbar(x_sum, mean_y2,
                  yerr=[std_y2, std_y2], capsize=15,
                  fmt='s', markersize=13, ecolor='blue', markeredgecolor="black", color='blue', mew=1)
    axes.plot(x_sum, mean_y2, c='blue', linewidth=0.5)

    axes.set_xlim(list_xlim)
    axes.set_ylim(list_ylim)
    # axes.xaxis.grid(True, which='minor')
    axes.set_xscale('log')
    axes.set_xticks(list_set_xticks)
    axes.set_xticklabels(list_set_xticklabels, fontsize=16)

    axes.set_yticks(list_set_yticks)
    axes.set_yticklabels(list_set_yticklabels, fontsize=16)
    # axes.grid()
    axes.yaxis.set_minor_locator(MultipleLocator(val_ticks))
    fig.savefig(fname_save)
    plt.show()

if __name__ == '__main__':
    path_dir = '/media/mswym/PortableSSD/translucency/'
    num_obj = 11

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth', 'sphere']
    comb_list = list(itertools.combinations(np.linspace(0, len(list_objname) - 1, len(list_objname)), 2))

    list_num_latent = [2, 4, 8, 16, 32, 64, 128, 256]
    list_metric = ['arr_0', 'arr_1']

    for metric in list_metric:
        if metric=='arr_1':
            name_met = 'lpips'
        else:
            name_met = 'psnr'
        result_mean_metric_transfer = []
        result_mean_metric_nontransfer = []
        for num_latent in list_num_latent:

            fname_1_11 = path_dir + 'rdm_latent' + str(num_latent) + '_stim1_1_stim2_11.npz' #for 1~10
            fname_2_22 = path_dir + 'rdm_latent' + str(num_latent) + '_stim1_2_stim2_22.npz' #for 11
            fname_1_21 = path_dir + 'rdm_latent' + str(num_latent) + '_stim1_1_stim2_21.npz' #for 1~10
            fname_2_12 = path_dir + 'rdm_latent' + str(num_latent) + '_stim1_2_stim2_12.npz' #for 1~10


            data_1_11 = np.load(fname_1_11)
            data_2_22 = np.load(fname_2_22)
            rdm_diag = []
            for ind in range(num_obj):
                if ind == num_obj-1:
                    rdm_diag.append(data_2_22[metric][ind][0])
                else:
                    rdm_diag.append(data_1_11[metric][ind+1][ind])

            data_1_21 = np.load(fname_1_21)
            data_1_21 = data_1_21[metric]
            data_2_12 = np.load(fname_2_12)
            data_2_12 = data_2_12[metric]

            rdm = np.zeros((num_obj,num_obj))
            for ind_row in range(num_obj):
                for ind_col in range(num_obj):
                    if ind_row == ind_col:
                        rdm[ind_row][ind_col] = rdm_diag[ind_row]
                    elif ind_row < ind_col:
                        rdm[ind_row][ind_col] = data_1_21[ind_row][ind_col]
                    elif ind_row > ind_col:
                        rdm[ind_row][ind_col] = data_2_12[ind_row][ind_col]

            fig = plt.figure(figsize=(6.5, 6.5))
            axes = fig.add_subplot(1, 1, 1)
            if metric == 'arr_0':
                axes.imshow(abs(100 - rdm), vmax=100, vmin=50, cmap='plasma')
            elif metric == 'arr_1':
                axes.imshow(rdm, vmax=0.4, vmin=0, cmap='plasma')
            fig.savefig(path_dir + 'fig/metric_' + name_met + 'rdm_latent' + str(num_latent))
            plt.show()
            plt.close()

            mean_encode_obj = []
            for ind_col in range(num_obj):
                tmp = copy.deepcopy(rdm[:][ind_col])
                tmp = np.delete(tmp,ind_col,0)
                mean_encode_obj.append(np.mean(tmp))

            result_mean_metric_transfer.append(mean_encode_obj)
            result_mean_metric_nontransfer.append(rdm_diag)

        result_mean_metric_transfer = np.array(result_mean_metric_transfer)
        result_mean_metric_nontransfer = np.array(result_mean_metric_nontransfer)
        if metric=='arr_1':
            fname_save_fig = path_dir + 'fig/metric_' + name_met + 'for_latentsummaryfigure.png'
            draw_distributions(np.mean(result_mean_metric_nontransfer, 1),
                               np.std(result_mean_metric_nontransfer, 1) / np.sqrt(11),
                               np.mean(result_mean_metric_transfer, 1),
                               np.std(result_mean_metric_transfer, 1) / np.sqrt(11), fname_save=fname_save_fig)
        else:
            fname_save_fig = path_dir + 'fig/metric_' + name_met + 'for_latentsummaryfigure.png'
            tmp_1 = 50-result_mean_metric_nontransfer
            tmp_2 = 50 - result_mean_metric_transfer
            draw_distributions(np.mean(tmp_1, 1),
                               np.std(tmp_1, 1) / np.sqrt(11),
                               np.mean(tmp_2, 1),
                               np.std(tmp_2, 1) / np.sqrt(11),
                               list_ylim=[0, 20],
                               list_set_xticklabels=['2', '4', '8', '16', '32', '64', '128', '256'],
                               list_set_xticks=[2, 4, 8, 16, 32, 64, 128, 256],
                               list_set_yticklabels=['0', '5', '10', '15', '20'],
                               list_set_yticks=[0, 5, 10, 15, 20],
                               val_ticks=1,
                               fname_save=fname_save_fig)

        fname_save = path_dir + 'fig/metric_' + name_met + 'for_latentsummaryfigure.npz'
        np.savez(fname_save, result_mean_metric_transfer, result_mean_metric_nontransfer)

    a = 1