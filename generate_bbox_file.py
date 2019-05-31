"""Generate text file containing labels and bounding box coordinates."""
import os

import numpy as np
from PIL import Image
from skimage import transform
import torch
from tqdm import tqdm
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


def create_path_to_index(analysis_res):
    assert 'res_paths' in analysis_res
    path_to_index = {path: i for i, path in
                     enumerate(analysis_res['res_paths'])}
    return path_to_index


def apply_preprocessing(mask,
                        path,
                        analysis_res,
                        processing,
                        path_to_index=None):
    """
    Apply pre-processing to mask.

    Args:
        mask: 4D torch.Tensor.
        path: String.
        analysis_res: dict.
        processing: String.
        path_to_index: dict.

    Returns:
        mask: 4D torch.Tensor.
    """
    if path_to_index is None:
        path_to_index = create_path_to_index(analysis_res)

    # If special pre-processing is desired, should expect an array
    # of masks.
    assert len(mask.shape) == 4

    # Check that the path is in the analysis_file.
    assert path in path_to_index

    # If the path hasn't been processed yet in analysis_file, don't apply any
    # special pre-processing.
    curr_index = path_to_index[path]
    if curr_index > analysis_res['i']:
        mask = torch.mean(mask, dim=0, keepdim=True)
        return mask

    # Get logit scores for current example when preserving and deleting with
    # masks.
    y_prevs = analysis_res['y_prevs'][curr_index]
    y_dels = analysis_res['y_dels'][curr_index]

    if 'crossover' in processing:
        # Get first index after crossover point.
        crossed_idx = np.where(y_prevs > y_dels)[0]
        if len(crossed_idx) == 0:
            crossed_index = mask.shape[0] - 1
        else:
            crossed_index = crossed_idx[0]

        # If using 'mean_crossover' pre-processing, take the mean of all
        # masks before and one immediately after the crossover point between
        # the logits of the preserved vs. deleted inputs.
        if processing == 'mean_crossover':
            mask = torch.mean(mask[:crossed_index + 1], dim=0, keepdim=True)
            return mask
        # If using 'single_crossover', take the first mask after the
        # crossover point.
        elif processing == 'single_crossover':
            mask = mask[crossed_index].unsqueeze(0)
            return mask
    else:
        assert False


def get_bbox_from_heatmap(heatmap, alpha, method='mean'):
    """Return bounding box coordinates for a thresholded heatmap.

    Args:
        heatmap: Numpy array, heatmap from which to threshold and extract
            a bounding box.
        alpha: Float, hyperparameter for thresholding.
        method: String, name of thresholding method.

    Returns:
        list containing bounding box coordinates.
    """
    # Threshold mask and get smallest bounding box coordinates.
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
        return [-1, -1, -1, -1]
    return [x[0],y[0],x[-1],y[-1]]


def generate_bbox_file(data_dir,
                       out_file,
                       method='mean',
                       alpha=0.5,
                       imdb_file='./data/val_imdb_0_1000.txt',
                       smooth=0.,
                       processing=None,
                       analysis_file=None):
    """

    Args:
        data_dir: String, directory containing torch results files.
        out_file: String, path to save output file.
        method: String, 'mean', 'min_max_diff', 'energy'
        alpha: Float, threshold value with which to threshold masks.
        imdb_file: String, path to imdb file.
        smooth: Float, amount of smoothing to apply.
        processing: String, type of pre-processing to apply to masks
            (i.e., 'mean_crossover', 'single_crossover', or None).
        analysis_file: String, path to analysis file, which contains
            information the result of applying masks to the input.
    """

    synsets = np.loadtxt('data/synsets.txt', dtype=str, delimiter='\t')

    imdb = np.loadtxt(imdb_file, dtype=str)
    
    imdb_val_ordered = np.loadtxt('./data/val.txt', dtype=str)
    image_names = imdb_val_ordered[:,0]

    paths = imdb[:,0]

    if processing is not None:
        # Check analysis file is given and exists.
        assert analysis_file is not None
        assert os.path.exists(analysis_file)

        # Load results onto CPU.
        analysis_res = torch.load(analysis_file,
                                  map_location=lambda storage, loc: storage)

        # Create mapping from path to index.
        path_to_index = create_path_to_index(analysis_res)

        # Check contents of analysis results.
        assert 'i' in analysis_res
        assert 'y_prevs' in analysis_res
        assert 'y_dels' in analysis_res
        assert 'y_origs' in analysis_res
        assert 'y_perturbs' in analysis_res

    idx = []
    bb_data = []
    for i, path in enumerate(tqdm(paths)):
        image_path = path
        image_name = path.split('/')[-1]
        synset = path.split('/')[-2]
        mask_path = os.path.join(data_dir, synset, image_name + '.pth')

        # Verify label and image name.
        assert(synset in synsets)
        assert(image_name in image_names)

        # Save index into ordered split.
        index = np.where(image_names == image_name)[0][0]
        idx.append(index)

        # Load results from torch file.
        if not os.path.exists(mask_path):
            print(f'{mask_path} does not exist.')
            bb_data.append([synset, -2, -2, -2, -2])
            continue

        res = torch.load(mask_path)

        # Get original image dimensions.
        img = Image.open(image_path)
        (img_w, img_h) = img.size

        # Load mask.
        mask = res['mask']

        # If no pre-processing, and mask is an array of masks, use the mean mask.
        if processing is None:
            if len(mask.shape) == 4:
                mask = torch.mean(mask, dim=0, keepdim=True)
        else:
            mask = apply_preprocessing(mask=mask,
                                       path=mask_path,
                                       analysis_res=analysis_res,
                                       processing=processing,
                                       path_to_index=path_to_index)

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
        bb = get_bbox_from_heatmap(heatmap=resized_mask,
                                   alpha=alpha,
                                   method=method)
        bb.insert(0, synset)
        bb_data.append(bb)

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
        parser.add_argument('--processing',
                            choices=['mean_crossover', 'single_crossover'],
                            default=None,
                            help='specify type of processing with which to '
                                 'apply to masks.')
        parser.add_argument('--analysis_file', type=str, default=None,
                            help='path of file containing information about '
                                 'the result of applying masks to input.')
        args = parser.parse_args()

        generate_bbox_file(data_dir=args.data_dir,
                           out_file=args.out_file,
                           method=args.method,
                           alpha=args.alpha,
                           imdb_file=args.imdb_file,
                           smooth=args.smooth,
                           processing=args.processing,
                           analysis_file=args.analysis_file)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
