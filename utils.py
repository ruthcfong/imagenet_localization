"""
utils.py

Contains helper functions.
"""

import os
import numpy as np


def str2bool(v):
    v = v.lower()
    if v in ('yes', 'true', 't', '1'):
        return True
    elif v in ('no', 'false', 'f', '0'):
        return False
    raise ValueError('Boolean argument needs to be true or false. '
                     'Instead, it is %s.' % v)


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


