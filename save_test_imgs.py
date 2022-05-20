import numpy as np
from translucency.class_mydataset import MyDatasetBinary
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from PIL import Image

def read_img_dataset(mypath, img_transform, batch_size_train):
    # load all image data
    dataset = MyDatasetBinary(mypath, transform1=img_transform, flag_hdr=True)
    dataset = DataLoader(dataset, batch_size=batch_size_train, shuffle=False)

    return dataset

def trans_cpu_and_save(x, dir_name):
    x = x.transpose(0, 2, 3, 1)
    for ind_img in range(x.shape[0]):
        tmp = 255 * np.reshape(x[ind_img], [x.shape[1], x.shape[2],x.shape[3]])
        tmp = Image.fromarray(np.uint8(tmp))

        tmp.save(dir_name + '/' + str(ind_img) + '.png', format='PNG')

if __name__ == '__main__':
    batch_size_test = 30
    size_input = np.array([256, 256, 3])
    path_dir_save = '/media/mswym/PortableSSD/translucency/'

    list_objname = ['armadillo', 'buddha', 'bun', 'bunny', 'bust', 'cap', 'cube', 'dragon', 'lucy', 'star_smooth',
                    'sphere']

    for ind_obj in list_objname:
        mypath_1 = path_dir_save + 'model_objects_tonemap_mask/che_220322_300test_' + ind_obj + '.binary'

        img_transform = transforms.Compose([
            transforms.Resize((size_input[0], size_input[1])),
            transforms.ToTensor()
        ])

        dir_name = path_dir_save + '_results/' + ind_obj
        os.makedirs(dir_name, exist_ok=True)

        img_test = read_img_dataset(mypath_1, img_transform, batch_size_test)
        img = []
        for data in img_test:
            tmp, _ = data
            tmp = tmp.to('cpu').detach().numpy().copy()
            img.append(tmp)
        img = np.array(img)
        img = img.reshape([-1, img.shape[2], img.shape[3], img.shape[4]])
        trans_cpu_and_save(img, dir_name)
