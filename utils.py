"""
utils.py

Contains helper functions.
"""

import math
import os
import numpy as np

import torch
import torch.nn.functional as F


EPSILON_DOUBLE = torch.tensor(2.220446049250313e-16, dtype=torch.float64)
EPSILON_SINGLE = torch.tensor(1.19209290E-07, dtype=torch.float32)
SQRT_TWO_DOUBLE = torch.tensor(math.sqrt(2), dtype=torch.float32)
SQRT_TWO_SINGLE = SQRT_TWO_DOUBLE.to(torch.float32)


def str2bool(v):
    v = v.lower()
    if v in ('yes', 'true', 't', '1'):
        return True
    elif v in ('no', 'false', 'f', '0'):
        return False
    raise ValueError('Boolean argument needs to be true or false. '
                     'Instead, it is %s.' % v)

def get_basename_without_ext(paths):
    """Return array of base names."""
    return np.array([os.path.splitext(os.path.basename(p))[0] for p in paths])


def create_dir_if_necessary(path, is_dir=False):
    """Create directory to path if necessary."""
    parent_dir = get_parent_dir(path) if not is_dir else path
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def get_parent_dir(path):
    """Return parent directory of path."""
    return os.path.abspath(os.path.join(path, os.pardir))


def write_imdb(split_dir, gt_file, out_file):
    """
    Write an imdb file containing paths and labels.

    Args:
        split_dir: String, path to image directory for dataset split.
        gt_file: String, path to ground truth file with relative image names.
        out_file: String, path of output file.
    """
    f = open(gt_file)
    rel_paths = []
    labels = []
    for line in f.readlines():
        s = line.split()
        rel_paths.append(s[0])
        labels.append(int(s[1]))

    abs_paths = []
    for p in rel_paths:
        abs_paths.append(os.path.join(split_dir, p))

    out_f = open(out_file, 'w')
    for i in range(len(abs_paths)):
        out_f.write('%s %d\n' % (abs_paths[i], labels[i]))
    out_f.close()


def read_imdb(imdb_file):
    """Load (paths, labels) from imdb file."""
    f = open(imdb_file)
    paths = []
    labels = []
    for line in f.readlines():
        s = line.split()
        paths.append(s[0])
        labels.append(int(s[1]))

    return (np.array(paths), np.array(labels))


def save_dummy_bbox_file(out_file = 'data/dummy_bb_file.txt'):
    """Save a dummy bounding box file."""
    _, labels = read_imdb('data/val.txt')
    num_examples = len(labels)
    synsets = np.loadtxt('data/synsets.txt', dtype='str', delimiter='\t')
    labels_as_synsets = np.array([synsets[l] for l in labels])[:, None]
    assert(labels_as_synsets.shape[0] == num_examples)
    bbs = np.random.randint(50, 100, size=(num_examples, 2))
    deltas = np.random.randint(1, 100, size=(num_examples, 2))
    labels_and_bbs = np.hstack((labels_as_synsets,
                                bbs[:, 0, None],
                                bbs[:, 1, None],
                                bbs[:, 0, None] + deltas[:, 0, None],
                                bbs[:, 1, None] + deltas[:, 0, None]))
    np.savetxt(out_file, labels_and_bbs, fmt='%s %s %s %s %s')


def imsmooth(x, sigma, stride=1, padding=0, padding_mode='constant', padding_value=0):
    r"""Apply a Gaussian filter to a batch of 2D images

    Args:
        x (Tensor): :math:`N\times C\times H\times W` image tensor.
        sigma (float): standard deviation of the Gaussian kernel.
        stride (int, optional): subsampling factor (default: 1).
        padding (int, optional): extra padding (default: 0).
        padding_mode (str, optional): `constant`, `reflect` or `replicate` (default: `constant`).
        padding_value (float, optional): constant value for the `constant` padding mode (default: 0).

    Returns:
        Tensor: :math:`N\times C\times H\times W` tensor with the smoothed images.
    """
    assert sigma >= 0
    W = math.ceil(4 * sigma)
    filt = torch.arange(-W, W+1, dtype=torch.float32, device=x.device) / \
           (SQRT_TWO_SINGLE * sigma + EPSILON_SINGLE)
    filt = torch.exp(- filt*filt)
    filt /= torch.sum(filt)
    num_channels = x.shape[1]
    W = W + padding
    if padding_mode == 'constant' and padding_value == 0:
        P = W
        y = x
    else:
        # pad: (before, after) pairs starting from last dimension backward
        y = F.pad(x, (W, W, W, W), mode=padding_mode, value=padding_value)
        P = 0
        padding = 0
    y = F.conv2d(y, filt.reshape((1, 1, -1, 1)).expand(num_channels, -1, -1, -1),
                 padding=(P, padding), stride=(stride, 1), groups=num_channels)
    y = F.conv2d(y, filt.reshape((1, 1, 1, -1)).expand(num_channels, -1, -1, -1),
                 padding=(padding, P), stride=(1, stride), groups=num_channels)
    return y


def normalize(arr):
    """Normalize array values to be between 0 and 1
        Args:
            arr (numpy array): non-normalized array
        Return:
            normalized array
    """
    if (arr == 0).all() or (arr == 1).all():
        return arr
    min_value = np.min(arr)
    max_value = np.max(arr)
    norm_arr = (arr - min_value) / (max_value - min_value)
    return norm_arr
