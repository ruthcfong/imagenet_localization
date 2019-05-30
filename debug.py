import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image
from skimage import transform
import torch
import torch.nn.functional as F
from utils import imsmooth, normalize, str2bool

data_dirs = {}
data_dirs['gradient'] = '/scratch/shared/slow/mandela/gradient'
data_dirs['guided_backprop'] ='/scratch/shared/slow/mandela/guided_backprop'
data_dirs['rise'] = '/scratch/shared/slow/mandela/rise'
data_dirs['grad_cam'] = '/scratch/shared/slow/mandela/grad_cam'
data_dirs['perturbations'] = '/scratch/shared/slow/vedaldi/vis/exp20-sal-im12val-vgg16'

alphas = {}
alphas['gradient'] = {'mean': 1.5, 'min_max_diff': 0.2, 'energy': 0.45}
alphas['guided_backprop'] = {'mean': 2.0, 'min_max_diff': 0.05, 'energy': 0.6}
alphas['rise'] = {'mean': 1.0, 'min_max_diff': 0.65, 'energy': 0.1}
alphas['grad_cam'] = {'mean': 2.0, 'min_max_diff': 0.3, 'energy': 0.4}
alphas['perturbations'] = {'mean': 1.0, 'min_max_diff': 0.5, 'energy': 0.55}


def generate_bbox(data_dir, image_path, synset, image_name, 
    method='mean', alpha=0.5, smooth=True):

    mask_path = os.path.join(data_dir, synset, image_name + '.pth')

    # Load results from torch file.
    if not os.path.exists(mask_path):
        print('DON EXIST')
        return [synset, -2, -2, -2, -2], None, None


    res = torch.load(mask_path)

    # Get original image dimensions.
    img = Image.open(image_path)
    (img_w, img_h) = img.size

    # Load and verify 2D mask.
    mask = res['mask']
    # if list of masks, find mean mask
    if len(mask.shape) == 4:
        mask = torch.mean(mask, dim=0, keepdim=True)       

    if smooth:
        mask = imsmooth(mask, sigma=20)

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
        heatmap = heatmap >= threshold
    elif method == 'min_max_diff':
        threshold = alpha*(heatmap.max()-heatmap.min())
        heatmap_m = heatmap - heatmap.min()
        heatmap = heatmap_m >= threshold
        heatmap[heatmap_m < threshold] = 0
    elif method == 'energy':
        heatmap_f = heatmap.flatten()
        sorted_idx = np.argsort(heatmap_f)[::-1]
        tot_energy = heatmap.sum()
        heatmap_cum = np.cumsum(heatmap_f[sorted_idx])
        ind = np.where(heatmap_cum >= alpha*tot_energy)[0][0]
        heatmap_f = np.ones(heatmap_f.shape)
        heatmap_f[sorted_idx[ind:]] = 0
        heatmap = np.reshape(heatmap_f, heatmap.shape)

    x = np.where(heatmap.sum(0) > 0)[0] + 1
    y = np.where(heatmap.sum(1) > 0)[0] + 1
    if len(x) == 0 or len(y) == 0:
        return [synset, -1, -1, -1, -1], resized_mask, heatmap
    return [synset, x[0],y[0],x[-1],y[-1]], resized_mask, heatmap


def compute_overlap(bb, bbgt):
    assert(len(bbgt) == 4)
    assert(len(bb) == 4)

    ov_vector = []
    bi=[max(bb[0],bbgt[0]), max(bb[1],bbgt[1]),min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
    iw=bi[2]-bi[0]+1
    ih=bi[3]-bi[1]+1
    ov = -1
    if iw>0 and ih>0:
    # Compute overlap as area of intersection / area of union.
        ua=((bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih)
        ov=iw*ih/float(ua)
        return ov
    else:
        return 0


def debug(method='min_max_diff', indexes = [25, 100, 346, 75, 100], save_path='.'):
    file_paths = np.loadtxt('./data/val_imdb_0_1000.txt', dtype=str)[indexes, 0]
    for file_path in file_paths:
        image = Image.open(file_path)
        image_name = file_path.split('/')[-1]
        filename = file_path.split('/')[-1].strip('.JPEG')
        synset = file_path.split('/')[-2]
        bboxes = None
        with open('/scratch/shared/slow/ruthfong/ILSVRC2012/val.csv') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                if row["fname"] == filename:
                    xmin = int(row["xmin"])
                    ymin = int(row["ymin"])
                    xmax = int(row["xmax"])
                    ymax = int(row["ymax"])
                    bbgt = (xmin, ymin, xmax, ymax)
                    resize_shape = (int(row["height"]), int(row["width"]))
                    break

        f, ax = plt.subplots(3, 5, figsize=(5*4, 3*4))
        plt.title(method)
        ax[0][0].set_ylabel('mask')
        ax[1][0].set_ylabel('heatmap (thres)')
        ax[2][0].set_ylabel('BBox')

        for i, mask_method in enumerate(['rise', 'guided_backprop', 'gradient', 'grad_cam', 'perturbations']):
            data_dir = data_dirs[mask_method]
            alpha = alphas[mask_method][method]
            smooth = True if mask_method == 'perturbations' else False
            bb_pred, mask, heatmap = generate_bbox(data_dir, file_path, synset, image_name, 
                method=method, alpha=alpha, smooth=smooth)
            if mask is None:
                continue
            overlap = compute_overlap(bb_pred[1:], bbgt)
            ax[0][i].imshow(mask, vmin=0, vmax=1, cmap='jet')
            ax[0][i].set_title(mask_method)
            ax[1][i].imshow(heatmap, vmin=0, vmax=1, cmap='jet')
            (xmin, ymin, xmax, ymax) = bb_pred[1:]
            (xmin_gt, ymin_gt, xmax_gt, ymax_gt) = bbgt
            ax[2][i].imshow(image)
            rect_gt = patches.Rectangle((xmin_gt, ymin_gt), xmax_gt-xmin_gt, 
                ymax_gt-ymin_gt, linewidth=1, edgecolor='r',facecolor='none')
            rect_p = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                linewidth=1, edgecolor='b',facecolor='none')
            ax[2][i].add_patch(rect_gt)
            ax[2][i].add_patch(rect_p)
            ax[2][i].set_title("Overlap: %.3f" % (overlap))
            ax[2][i].legend((rect_gt, rect_p), ('Ground Truth', 'Predicted'))
        plt.savefig(os.path.join(save_path, filename + '_' + method))
        plt.close()        


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', str2bool)
        parser.add_argument('--method', type=str, default='min_max_diff')
        parser.add_argument('--save_path', type=str, default='.')
        parser.add_argument('--indexes', type=int, nargs='*', default=[25, 100, 346, 75, 100])

        args = parser.parse_args()
    
        debug(method=args.method, indexes=args.indexes, save_path=args.save_path)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)