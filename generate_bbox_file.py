"""Generate text file containing labels and bounding box coordinates."""
import os

import numpy as np
from PIL import Image
from skimage import transform
import torch
from utils import imsmooth

VALID_SPLITS = ['val']


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


def generate_bbox_file(data_dir,
                       out_file,
                       method='mean',
                       alpha=0.5,
                       imdb_file='./data/val_imdb_0_1000.txt',
                       smooth=0.):
    """

    Args:
        data_dir: String, directory containing torch results files.
        out_file: String, path to save output file.
        method: String, 'mean', 'min_max_diff', 'energy'
        alpha: Float, threshold value with which to threshold masks.
        imdb_file: String, path to imdb file.
        smooth: Float, amount of smoothing to apply.
    """

    synsets = np.loadtxt('data/synsets.txt', dtype=str, delimiter='\t')

    imdb = np.loadtxt(imdb_file, dtype=str)
    
    imdb_val_ordered = np.loadtxt('./data/val.txt', dtype=str)
    image_names = imdb_val_ordered[:,0]

    paths = imdb[:,0]

    idx = []
    bb_data = []
    for i, path in enumerate(paths):
        image_path = path
        image_name = path.split('/')[-1]
        synset = path.split('/')[-2]
        mask_path = os.path.join(data_dir, synset, image_name + '.pth')
        print(i)

        # Verify label and image name.
        assert(synset in synsets)
        assert(image_name in image_names)

        # Save index into ordered split.
        index = np.where(image_names == image_name)[0][0]
        idx.append(index)

        # Load results from torch file.
        if not os.path.exists(mask_path):
            print('DON EXIST')
            bb_data.append([synset, -2, -2, -2, -2])
            continue

        res = torch.load(mask_path)

        # Get original image dimensions.
        img = Image.open(image_path)
        (img_w, img_h) = img.size

        # Load and verify 2D mask.
        mask = res['mask']
        # if list of masks, find mean mask
        if len(mask.shape) == 4:
            mask = torch.mean(mask, dim=0, keepdim=True)       

        # Apply smoothing to heatmap.
        if smooth > 0.:
            mask = imsmooth(mask, sigma=smooth)

        mask = mask.squeeze()
        assert(len(mask.shape) == 2)

        mask = mask.cpu().data.squeeze().numpy()

        if (not np.max(mask) <= 1) or (not np.min(mask) >= 0):
            print('Normalizing')
            mask = normalize(mask)
        assert(np.max(mask) <= 1)
        assert(np.min(mask) >= 0)

        # Resize mask to original image dimensions.
        resized_mask = transform.resize(mask, (img_h, img_w))
        assert(np.max(resized_mask) <= 1)
        assert(np.min(resized_mask) >= 0)

        # Threshold mask and get smallest bounding box coordinates.
        heatmap = resized_mask
        if method == 'mean':
            threshold = alpha*heatmap.mean()
            heatmap[heatmap < threshold] = 0
        elif method == 'min_max_diff':
            threshold = alpha*(heatmap.max()-heatmap.min())
            heatmap_m = heatmap - heatmap.min()
            heatmap[heatmap_m < threshold] = 0
        elif method == 'energy':
            heatmap_f = heatmap.flatten()
            sorted_idx = np.argsort(heatmap_f)[::-1]
            tot_energy = heatmap.sum()
            heatmap_cum = np.cumsum(heatmap_f[sorted_idx])
            ind = np.where(heatmap_cum >= alpha*tot_energy)[0][0]
            heatmap_f[sorted_idx[ind:]] = 0
            heatmap = np.reshape(heatmap_f, heatmap.shape)
        elif method == 'threshold':
            threshold = alpha
            heatmap[heatmap < threshold] = 0

        x = np.where(heatmap.sum(0) > 0)[0] + 1
        y = np.where(heatmap.sum(1) > 0)[0] + 1
        if len(x) == 0 or len(y) == 0:
            bb_data.append([synset, -1, -1, -1, -1]) 
            continue
        bb_data.append([synset, x[0],y[0],x[-1],y[-1]])

    bb_data = np.array(bb_data)
    idx = np.array(idx)

    # Save bounding box information in correct order.
    sorted_idx = np.argsort(idx)
    np.savetxt(out_file, bb_data, fmt='%s %s %s %s %s')


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('data_dir', type=str)
        parser.add_argument('out_file', type=str)
        parser.add_argument('--method', type=str, default='mean')
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--imdb_file', type=str, default='./data/val_imdb_0_1000.txt')
        parser.add_argument('--smooth', type=float, default=0.,
                            help='sigma for smoothing to apply to heatmap '
                                 '(default: 0.).')
        args = parser.parse_args()

        generate_bbox_file(data_dir=args.data_dir,
                           out_file=args.out_file,
                           alpha=args.alpha,
                           imdb_file=args.imdb_file,
                           smooth=args.smooth)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
