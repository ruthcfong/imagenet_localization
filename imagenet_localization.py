import argparse
import os
import sys
import time

import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import resize
from sklearn.metrics import average_precision_score

from compute_localization_results import load_objs
from utils import imsmooth


def compute_localization_ap(exp_dir, image_dir, annotation_dir, out_file, figures_dir=None):
    # Get synset information.
    synset_words = np.loadtxt(
        '/users/ruthfong/pytorch_workflow/synset_words.txt', dtype=str,
        delimiter='\t')
    synset2label = {w.split(' ')[0]: ' '.join(w.split(',')[0].split(' ')[1:])
                    for w in synset_words}

    if figures_dir is not None and not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    sub_dirs = np.sort([os.path.join(exp_dir, f) for f in os.listdir(exp_dir)])
    res = {}
    for j, class_dir in enumerate(sub_dirs):
        synset = os.path.basename(class_dir)
        res_files = np.sort(
            [os.path.join(class_dir, f) for f in os.listdir(class_dir) if
             'pth' in f])
        aps = np.zeros(len(res_files))
        start = time.time()

        if figures_dir is not None:
            f, ax = plt.subplots(5, 10, fig_size=(10*4, 5*4))

        for i, res_file in enumerate(res_files):
            anno_file = os.path.join(annotation_dir,
                                     os.path.basename(res_file).strip(
                                         '.JPEG.pth') + '.xml')
            objs = load_objs(anno_file)

            img_path = os.path.join(image_dir, synset,
                                    os.path.basename(res_file).strip('.pth'))
            img = Image.open(img_path).convert('RGB')

            (w, h) = img.size

            mask = np.zeros((h, w))

            for obj, bbs in objs.items():
                for bb in bbs:
                    x0, y0, x1, y1 = bb
                    mask[y0:y1, x0:x1] = 1.

            x = torch.load(res_file)

            vis = resize(
                imsmooth(torch.sum(x['mask'], 0, keepdim=True),
                         sigma=20).squeeze().cpu().data.numpy(),
                (h, w))
            y_flat = mask.reshape(-1)
            vis_flat = vis.reshape(-1)
            ap = average_precision_score(y_flat, vis_flat)
            aps[i] = ap

            if figures_dir is not None:
                a = ax[int(i/10)][int(i%10)]
                a.imshow(img)
                a.imshow(vis, alpha=0.5, cmap='jet')
                a.imshow(mask, alpha=0.5)
                a.set_title(f"{aps[i]:.2f}")
                a.set_xticks([])
                a.set_yticks([])

        if figures_dir is not None:
            plt.show()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"{synset}_{synset2label[synset]}.pdf"), format='pdf')
            plt.close()

        print(f"[{j+1}/{len(sub_dirs)}]\t{synset} ({synset2label[synset]})\t"
              f"{np.mean(aps):2f}\t{time.time() - start:2f} secs")

        res[synset] = {
            'map': np.mean(aps),
            'aps': aps,
            'file_list': res_files,
            'label': synset2label[synset],
        }

    torch.save(res, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_file', type=str)
    parser.add_argument('--exp_dir', type=str,
                        default='/scratch/shared/slow/vedaldi/vis/exp20-sal-im12val-vgg16')
    parser.add_argument('--image_dir', type=str,
                        default='/scratch/shared/slow/ruthfong/ILSVRC2012/images/val_pytorch/')
    parser.add_argument('--annotation_dir', type=str,
                        default='/datasets/imagenet14/cls_loc/val')
    parser.add_argument('--figures_dir', type=str, default=None)

    args = parser.parse_args()
    compute_localization_ap(exp_dir=args.exp_dir,
                            image_dir=args.image_dir,
                            annotation_dir=args.annotation_dir,
                            out_file=args.out_file,
                            figures_dir=args.figures_dir)

