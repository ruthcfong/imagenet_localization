"""Generate text file containing labels and bounding box coordinates."""
import os

import numpy as np
from PIL import Image
from skimage import transform
import torch

VALID_SPLITS = ['val']


def generate_bbox_file(data_dir, out_file, threshold=0.5, split='val'):
    """

    Args:
        data_dir: String, directory containing torch results files.
        out_file: String, path to save output file.
        threshold: Float, threshold value with which to threshold masks.
        split: String, name of ImageNet split.
    """
    assert(split in VALID_SPLITS)

    synsets = np.loadtxt('data/synsets.txt', dtype=str, delimiter='\t')

    imdb = np.loadtxt('data/%s.txt' % split, dtype=str)
    image_names = imdb[:,0]

    # Get paths to torch files.
    paths = np.sort([os.path.join(data_dir, f) for f in os.listdir(data_dir)])

    idx = []
    bb_data = []
    for i, path in enumerate(paths):
        # Load results from torch file.
        res = torch.load(path)

        # Get image name and label (synset).
        image_path = res['image_url']
        image_name = image_path.split('/')[-1]
        synset = image_path.split('/')[-2]

        # Verify label and image name.
        assert(synset in synsets)
        assert(image_name in image_names)

        # Save index into ordered split.
        index = np.where(image_names == image_name)[0][0]
        idx.append(index)

        # Get original image dimensions.
        img = Image.open(image_path)
        (img_w, img_h) = img.size

        # Load and verify 2D mask.
        mask = res['mask'].cpu().data.squeeze().numpy()
        assert(len(mask.shape) == 2)
        assert(np.max(mask) <= 1)
        assert(np.min(mask) >= 0)

        # Resize mask to original image dimensions.
        resized_mask = transform.resize(mask, (img_h, img_w))
        assert(np.max(resized_mask) <= 1)
        assert(np.min(resized_mask) >= 0)

        # Threshold mask and get smallest bounding box coordinates.
        ys, xs = np.where(resized_mask > threshold)
        xs += 1
        ys += 1
        x_min = np.min(xs)
        x_max = np.max(xs)
        y_min = np.min(ys)
        y_max = np.max(ys)

        # Save label and bounding box coordinates.
        bb_data.append([synset, x_min, y_min, x_max, y_max])

    bb_data = np.array(bb_data)
    idx = np.array(idx)

    # Save bounding box information in correct order.
    sorted_idx = np.argsort(idx)
    np.savetxt(out_file, bb_data[sorted_idx], fmt='%s %s %s %s %s')


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('data_dir', type=str)
        parser.add_argument('out_file', type=str)
        parser.add_argument('--threshold', type=float, default=0.5)
        parser.add_argument('--split', choices=VALID_SPLITS, default='val')
        args = parser.parse_args()

        generate_bbox_file(data_dir=args.data_dir,
                           out_file=args.out_file,
                           threshold=args.threshold,
                           split=args.split)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
