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


# TODO(ruthfong): Refactor with attribution/analyze_utils.py function.
def get_res_paths_dict(data_dir):
    res_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_dir)
                 for f in filenames if '.pth' in f]
    res_paths_d = {}
    for res_path in res_paths:
        img_name = os.path.basename(res_path)
        if '-class' in img_name:
            img_name, _ = img_name.split('-class')
        assert img_name not in res_paths_d
        res_paths_d[img_name] = res_path
    return res_paths_d


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
        # TODO(ruthfong): Note -- you can use "threshold" instead of "min_max_diff" if already normalized.
        threshold = alpha*(heatmap.max()-heatmap.min())
        heatmap_m = heatmap - heatmap.min()
        heatmap[heatmap_m < threshold] = 0
    elif method == 'energy':
        # TODO(ruthfong): Speed up "energy" for multiple alphas.
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
                       image_dir,
                       method='mean',
                       alpha=0.5,
                       imdb_file='./data/val_imdb_pytorch.txt',
                       smooth=0.,
                       processing=None,
                       analysis_file=None,
                       first_n=None):
    """

    Args:
        data_dir: String, directory containing torch results files.
        out_file: String, path to save output file.
        image_dir; String, directory containing image files.
        method: String, 'mean', 'min_max_diff', 'energy'
        alpha: Float, list, or np.ndarray, threshold value(s) with which to
            threshold masks.
        imdb_file: String, path to imdb file.
        smooth: Float, amount of smoothing to apply.
        processing: String, type of pre-processing to apply to masks
            (i.e., 'mean_crossover', 'single_crossover', or None).
        analysis_file: String, path to analysis file, which contains
            information the result of applying masks to the input.
        first_n: Integer, only generate bounding box information for
            first N examples.
    """

    synsets = np.loadtxt('data/synsets.txt', dtype=str, delimiter='\t')

    imdb = np.loadtxt(imdb_file, dtype=str)
    
    imdb_val_ordered = np.loadtxt('./data/val.txt', dtype=str)
    image_names = imdb_val_ordered[:,0]

    # Get image paths.
    image_paths = [os.path.join(image_dir, f) for f in imdb[:,0]]

    # Get dictionary mapping image names (without extension) to res paths.
    res_paths_lookup = get_res_paths_dict(data_dir)

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

    # Initialize variable for saving bounding boxes.
    if isinstance(alpha, float):
        bb_data = []
        print(f'Using alpha: {alpha}')
    elif isinstance(alpha, list) or isinstance(alpha, np.ndarray):
        bb_data = [[] for _ in range(len(alpha))]
        print(f'Using alpha range: {alpha}')
    else:
        assert False

    for i, image_path in enumerate(tqdm(image_paths)):
        image_name = os.path.basename(image_path)
        # TODO(ruthfong): Make getting synset more robust.
        synset = image_path.split('/')[-2]

        if first_n is not None and i == first_n:
            break

        # Verify label and image name.
        assert(synset in synsets)
        assert(image_name in image_names)

        # Check if results file exists for image.
        image_name_no_ext = os.path.splitext(image_name)[0]
        if image_name_no_ext not in res_paths_lookup:
            print(f'Results file for {image_name} does not exist.')
            if isinstance(alpha, float):
                bb_data.append([synset, -2, -2, -2, -2])
            else:
                for j in range(len(alpha)):
                    bb_data[j].append([synset, -2, -2, -2, -2])
            continue

        # Load results from torch file.
        mask_path = res_paths_lookup[image_name_no_ext]
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

        # Get bounding box for either a single or list of alpha choices.
        if isinstance(alpha, float):
            bb = get_bbox_from_heatmap(heatmap=resized_mask,
                                       alpha=alpha,
                                       method=method)
            bb.insert(0, synset)
            bb_data.append(bb)
        elif isinstance(alpha, list) or isinstance(alpha, np.ndarray):
            for j, a in enumerate(alpha):
                bb = get_bbox_from_heatmap(heatmap=resized_mask,
                                           alpha=a,
                                           method=method)
                bb.insert(0, synset)
                bb_data[j].append(bb)
        else:
            assert False


    # Save bounding box information.
    if isinstance(alpha, float):
        np.savetxt(out_file, np.array(bb_data), fmt='%s %s %s %s %s')
    else:
        for j, a in enumerate(alpha):
            prefix, ext = os.path.splitext(out_file)
            o_file = f'{prefix}_{a:.2f}{ext}'
            np.savetxt(o_file, np.array(bb_data[j]), fmt='%s %s %s %s %s')



if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('data_dir', type=str)
        parser.add_argument('out_file', type=str)
        parser.add_argument('--image_dir', type=str,
                            default='/scratch/shared/slow/ruthfong/ILSVRC2012/images/val_pytorch',
                            help='directory containing image files (PyTorch style).')
        parser.add_argument('--method',
                            type=str,
                            choices=['mean', 'min_max_diff', 'energy', 'threshold'],
                            default='mean')
        parser.add_argument('--alpha_range', action='store_true', default=False,
                            help='If True, use range of alpha.')
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--imdb_file', type=str, default='./data/val_imdb_pytorch.txt')
        parser.add_argument('--smooth', type=float, default=0.,
                            help='sigma for smoothing to apply to heatmap '
                                 '(default: 0.).')
        parser.add_argument('--processing',
                            choices=['mean_crossover', 'single_crossover'],
                            default=None,
                            help='specify type of processing with which to '
                                 'apply to masks.')
        parser.add_argument('--analysis_file', type=str,
                            default='/scratch/shared/slow/ruthfong/attribution/results/analyze/exp20-sal-im12val-vgg16.pth',
                            help='path of file containing information about '
                                 'the result of applying masks to input.')
        parser.add_argument('--first_n', type=int, default=None,
                            help='Only generate bounding box for first N examples.')
        args = parser.parse_args()

        if args.alpha_range:
            if args.method == "mean":
                alpha = np.arange(0, 10, 0.5)
            else:
                alpha = np.arange(0, 1, 0.05)
        else:
            alpha = args.alpha


        generate_bbox_file(data_dir=args.data_dir,
                           out_file=args.out_file,
                           image_dir=args.image_dir,
                           method=args.method,
                           alpha=alpha,
                           imdb_file=args.imdb_file,
                           smooth=args.smooth,
                           processing=args.processing,
                           analysis_file=args.analysis_file,
                           first_n=args.first_n)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
