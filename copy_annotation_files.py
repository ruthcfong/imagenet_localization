"""
copy_annotation_files.py

Given an imdb file containing relative image paths, copy those images'
annotation files from one directory to a new directory.
"""
import os
import shutil

from utils import read_imdb, create_dir_if_necessary


def copy_annotation_files(new_anno_dir, imdb_path, base_anno_dir):
    """

    Args:
        new_anno_dir: String, path to new directory to copy files to.
        imdb_path: String, path to imdb file containing relative image names
            and labels.
        base_anno_dir: String, path to original base directory containing
            annotation files.
    """
    # Get relative image paths.
    (rel_img_paths, _) = read_imdb(imdb_path)

    # For each image, copy its annotation file to the new directory.
    for i, rel_img_path in enumerate(rel_img_paths):
        rel_img_path_no_ext, _ = os.path.splitext(rel_img_path)
        rel_anno_path = '%s.xml' % rel_img_path_no_ext

        # Get original and new paths for the annotation file.
        orig_anno_path = os.path.join(base_anno_dir, rel_anno_path)
        new_anno_path = os.path.join(new_anno_dir,
                                     os.path.basename(rel_anno_path))

        # Create dirs as needed and copy the file.
        create_dir_if_necessary(new_anno_path)
        shutil.copyfile(orig_anno_path, new_anno_path)
        if i % 100 == 0:
            print('[%d/%d]' % (i, len(rel_img_paths)))


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--new_annotation_dir', type=str,
                            default='/scratch/shared/slow/ruthfong/imagenet14/cls_loc/annotated_train_heldout',
                            help='Path to dir to copy annotation files to.')
        parser.add_argument('--file_list', type=str,
                            default='data/annotated_train_heldout.txt',
                            help='Path to file containing relative image '
                                 'paths and labels.')
        parser.add_argument('--base_annotation_dir', type=str,
                            default='/datasets/imagenet14/cls_loc',
                            help='Path to dir with ImageNet annotations.')

        args = parser.parse_args()

        copy_annotation_files(args.new_annotation_dir,
                              imdb_path=args.file_list,
                              base_anno_dir=args.base_annotation_dir)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
