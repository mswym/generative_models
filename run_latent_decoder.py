import numpy as np

from translucency.vae_vanilla import VAE_vanilla
from translucency.class_mydataset import MyDatasetBinary
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.linear_model import LinearRegression

import time
import pickle
import matplotlib.pyplot as plt
import itertools

def make_decoder(x, y, scale_eigenval=1, autoadjust_eigenval=True):
    out = {}
    model_lr = LinearRegression(fit_intercept=True)
    model_lr.fit(x, y)
    eigen_val, eigen_vec = np.linalg.eig(model_lr.coef_)
    if autoadjust_eigenval:
        eigen_val = eigen_val / np.linalg.norm(eigen_val)
    else:
        eigen_val = eigen_val * scale_eigenval
    # eigen_val = np.ones(eigen_val.shape)
    eigen_val = np.diag(eigen_val)
    # for i in range(len(eigen_vec)):
    #    eigen_vec[i] = eigen_vec[i] / np.linalg.norm(eigen_vec[i])
    new_coef_matrix = np.dot(np.dot(eigen_vec, eigen_val), np.linalg.inv(eigen_vec))
    new_coef_matrix = np.real(new_coef_matrix)

    out['coefficient'] = new_coef_matrix
    out['intercept'] = model_lr.intercept_
    return out


def extract_latent(model, img_train):
    latent_z, _, _, _ = model._run_step(img_train)

    return latent_z.to('cpu').detach().numpy().copy()

def extract_latents(num_itr, model, path_checkpoint, img_train):
    model = model.load_from_checkpoint(checkpoint_path=path_checkpoint)
    model.eval()
    latent_z = [extract_latent(model, next(iter(img_train))[0]) for itr in range(10)]
    latent_z = np.array(latent_z)

    return latent_z.reshape([-1, latent_z.shape[2]])

def read_img_dataset(mypath, img_transform, batch_size_train):
    #load all image data
    dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)
    dataset = DataLoader(dataset, batch_size=batch_size_train, shuffle=False)

    return dataset


def cal_linear_reg(x, new_coef_matrix, intercept):
    x_hat = np.dot(new_coef_matrix, x.T)

    return x_hat.T + intercept

if __name__ == '__main__':
    batch_size_train = 30
    batch_size_test = 300
    size_input = np.array([256, 256, 3])
    scale_eigenval = 1
    flag_autoadjust_eigenval = False

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
                    'sphere']
    #list_objname = ['sphere', 'armadillo']
    list_num_latent = [2, 4, 8, 16, 32, 64, 128, 256]
    #list_num_latent = [256]

    path_dir_save = '/media/mswym/PortableSSD/translucency/'
    model_body = VAE_vanilla()

    comb_list = list(itertools.combinations(np.linspace(0, len(list_objname) - 1, len(list_objname)), 2))

    for num_latent in list_num_latent:
        for ind_obj in range(len(comb_list)):
            start_time = time.perf_counter()
            ind_obj_1 = int(comb_list[ind_obj][0])
            ind_obj_2 = int(comb_list[ind_obj][1])

            path_save_1 = path_dir_save + "obj_mask/coef/coef_" + list_objname[ind_obj_1] + "-" + list_objname[ind_obj_2] + "_latent" + str(num_latent) + ".pkl"
            path_save_2 = path_dir_save + "obj_mask/coef/coef_" + list_objname[ind_obj_2] + "-" + list_objname[ind_obj_1] + "_latent" + str(num_latent) + ".pkl"

            path_checkpoint_1 = path_dir_save + "obj_mask/" + list_objname[
                ind_obj_1] + "_latent" + str(num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"
            path_checkpoint_2 = path_dir_save + "obj_mask/" + list_objname[
                ind_obj_2] + "_latent" + str(num_latent) + "_logs/version_0/checkpoints/epoch=99-step=8499.ckpt"

            mypath_1 = path_dir_save + 'model_objects_tonemap_mask/che_220322_1500train_' + list_objname[
                ind_obj_1] + '.binary'
            mypath_2 = path_dir_save + 'model_objects_tonemap_mask/che_220322_1500train_' + list_objname[
                ind_obj_2] + '.binary'

            img_transform = transforms.Compose([
                transforms.Resize((size_input[0], size_input[1])),
                transforms.ToTensor()
            ])

            img_train_1 = read_img_dataset(mypath_1, img_transform, batch_size_train)
            img_train_2 = read_img_dataset(mypath_2, img_transform, batch_size_train)

            #latent computing.
            latent_z_1 = extract_latents(len(img_train_1), model_body, path_checkpoint_1, img_train_1)
            latent_z_2 = extract_latents(len(img_train_2), model_body, path_checkpoint_2, img_train_2)

            # train the linear decoder
            decoder_12 = make_decoder(latent_z_1, latent_z_2, scale_eigenval,
                                                            autoadjust_eigenval=flag_autoadjust_eigenval)
            decoder_21 = make_decoder(latent_z_2, latent_z_1, scale_eigenval,
                                                            autoadjust_eigenval=flag_autoadjust_eigenval)

            with open(path_save_1, 'wb') as f:
                pickle.dump(decoder_12, f)
            with open(path_save_2, 'wb') as f:
                pickle.dump(decoder_21, f)
            print(time.perf_counter()-start_time)

            #latent_z_12 = cal_linear_reg(latent_z_1, decoder_12['coefficient'], decoder_12['intercept'])
            #latent_z_21 = cal_linear_reg(latent_z_2, decoder_21['coefficient'], decoder_21['intercept'])
            #plt.scatter(latent_z_21.flatten(),latent_z_1.flatten())
            #plt.show()
            #plt.scatter(latent_z_12.flatten(), latent_z_2.flatten())
            #plt.show()
    print('finished')

