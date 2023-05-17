import os
import numpy as np
import scipy.io

from skimage.transform import resize
from tqdm import tqdm


def preprocess_boe_chiu_dataset(folder_path, width, height, width_out, height_out):
    subject_path = [os.path.join(folder_path, 'Subject_0{}.mat'.format(i)) for i in range(1, 10)]\
            + [os.path.join(folder_path, 'Subject_10.mat')]

    # only these samples have mask labels
    data_indexes = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]

    def thresh(x):
        if x == 0:
            return 0
        else:
            return 1
    thresh = np.vectorize(thresh, otypes=[np.float32])

    def process_array(img_tensor, fluid_tensor, skip_img=False):
        img_array = None
        if not skip_img:
            img_array = np.transpose(img_tensor, (2, 0, 1)) / 255
            img_array = resize(img_array, (img_array.shape[0], width, height))

        fluid_array = np.transpose(fluid_tensor, (2, 0, 1))
        fluid_array = np.nan_to_num(fluid_array)
        fluid_array = resize(fluid_array, (fluid_array .shape[0], width_out, height_out))
        fluid_array = thresh(fluid_array)  # convert to hard mask
        return img_array, fluid_array

    def create_dataset(paths):
        x = []
        y_manual_1 = []
        y_manual_2 = []
        y_auto = []

        for path in tqdm(paths):
            mat = scipy.io.loadmat(path)
            img_tensor = mat['images']
            fluid_tensor_1 = mat['manualFluid1']
            fluid_tensor_2 = mat['manualFluid2']
            fluid_tensor_auto = mat['automaticFluidDME']

            img_array, fluid_array_1 = process_array(img_tensor, fluid_tensor_1)
            _, fluid_array_2 = process_array(img_tensor, fluid_tensor_2)
            _, fluid_array_auto = process_array(img_tensor, fluid_tensor_auto)

            for idx in data_indexes:
                x += [np.expand_dims(img_array[idx], 0)]
                y_manual_1 += [np.expand_dims(fluid_array_1[idx], 0)]
                y_manual_2 += [np.expand_dims(fluid_array_2[idx], 0)]
                y_auto += [np.expand_dims(fluid_array_auto[idx], 0)]

        dataset = {
            'x': np.array(x),
            'y_manual_1': np.array(y_manual_1),
            'y_manual_2': np.array(y_manual_2),
            'y_auto': np.array(y_auto),
        }

        # create non zero mask
        mask_dict = {}
        for k, v in dataset.items():
            if k.startswith('y'):
                non_zero_mask = np.sum(v, axis=(1, 2, 3)) != 0
                mask_dict[f'{k}_mask'] = non_zero_mask
        dataset.update(mask_dict)
        return dataset

    train_dataset = create_dataset(subject_path[:9])
    test_dataset = create_dataset(subject_path[9:])

    # save processed dataset
    np.savez(f'{folder_path}/train_dataset', **train_dataset)
    np.savez(f'{folder_path}/test_dataset', **test_dataset)


if __name__ == '__main__':
    width = 284
    height = 284
    width_out = 196
    height_out = 196
    preprocess_boe_chiu_dataset(
        'data/2015_boe_chiu/2015_BOE_Chiu',
        width,
        height,
        width_out,
        height_out
    )
