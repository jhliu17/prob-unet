import itertools
import numpy as np
import torch

from matplotlib import pyplot as plt
from torch.utils.data import random_split


def check_attributes(object_, attributes):

    missing = []
    for attr in attributes:
        if not hasattr(object_, attr):
            missing.append(attr)
    if len(missing) > 0:
        return False
    else:
        return True


def set_seeds(seed, cuda=True):
    if not hasattr(seed, "__iter__"):
        seed = (seed, seed, seed)
    np.random.seed(seed[0])
    torch.manual_seed(seed[1])
    if cuda:
        torch.cuda.manual_seed_all(seed[2])


def make_onehot(array, labels=None, axis=1, newaxis=False):
    # get labels if necessary
    if labels is None:
        labels = np.unique(array)
        labels = list(map(lambda x: x.item(), labels))

    # get target shape
    new_shape = list(array.shape)
    if newaxis:
        new_shape.insert(axis, len(labels))
    else:
        new_shape[axis] = new_shape[axis] * len(labels)

    # make zero array
    if type(array) == np.ndarray:
        new_array = np.zeros(new_shape, dtype=array.dtype)
    elif torch.is_tensor(array):
        new_array = torch.zeros(new_shape, dtype=array.dtype, device=array.device)
    else:
        raise TypeError("Onehot conversion undefined for object of type {}".format(type(array)))

    # fill new array
    n_seg_channels = 1 if newaxis else array.shape[axis]
    for seg_channel in range(n_seg_channels):
        for l, label in enumerate(labels):
            new_slc = [slice(None), ] * len(new_shape)
            slc = [slice(None), ] * len(array.shape)
            new_slc[axis] = seg_channel * len(labels) + l
            if not newaxis:
                slc[axis] = seg_channel
            new_array[tuple(new_slc)] = array[tuple(slc)] == label

    return new_array


def match_to(x, ref, keep_axes=(1,)):

    target_shape = list(ref.shape)
    for i in keep_axes:
        target_shape[i] = x.shape[i]
    target_shape = tuple(target_shape)
    if x.shape == target_shape:
        pass
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 2:
        while x.dim() < len(target_shape):
            x = x.unsqueeze(-1)

    x = x.expand(*target_shape)
    x = x.to(device=ref.device, dtype=ref.dtype)

    return x


def make_slices(original_shape, patch_shape):

    working_shape = original_shape[-len(patch_shape):]
    splits = []
    for i in range(len(working_shape)):
        splits.append([])
        for j in range(working_shape[i] // patch_shape[i]):
            splits[i].append(slice(j*patch_shape[i], (j+1)*patch_shape[i]))
        rest = working_shape[i] % patch_shape[i]
        if rest > 0:
            splits[i].append(slice((j+1)*patch_shape[i], (j+1)*patch_shape[i] + rest))

    # now we have all slices for the individual dimensions
    # we need their combinatorial combinations
    slices = list(itertools.product(*splits))
    for i in range(len(slices)):
        slices[i] = [slice(None), ] * (len(original_shape) - len(patch_shape)) + list(slices[i])

    return slices


def plot_segmentation_examples(datax, datay_pred, datay, num_examples=3):
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(18, 4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        ax[row_num][0].imshow(np.transpose(datax[image_indx], (1, 2, 0))[:, :, 0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.transpose(datay_pred[image_indx], (1, 2, 0))[:, :, 0])
        ax[row_num][1].set_title("Segmented Image")
        ax[row_num][2].imshow(datay_pred[image_indx].argmax(0))
        ax[row_num][2].set_title("Segmented Image localization")
        ax[row_num][3].imshow(np.transpose(datay[image_indx], (1, 2, 0))[:, :, 0])
        ax[row_num][3].set_title("Target image")

    return fig, ax


def split_dataset(dataset, train_ratio, seed):
    train_size = int(len(dataset) * train_ratio)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(seed)
    )
    return train_dataset, valid_dataset
