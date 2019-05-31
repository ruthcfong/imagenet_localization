import numpy as np
import os 
import torch

from compute_localization_results import compute_localization_results
from generate_bbox_file import generate_bbox_file
from utils import str2bool


def find_best_alpha(
    imdb_file='./data/annotated_train_heldout_imdb.txt', 
    annotation_dir='/scratch/shared/slow/ruthfong/imagenet14/cls_loc/annotated_train_heldout',
    attribution_method='pertrubations',
    method='mean',
    alphas=np.arange(0,10,0.1),
    verbose=True,
    out_path='/scratch/shared/slow/mandela/bbox_results'
):
    errs = np.zeros(len(alphas))
    results = []
    overlaps = []
    num_blacklists = []
    no_masks = []
    for i in range(len(alphas)):
        print(i)
        alpha = alphas[i]
        bb_name = 'bb_val_%s_%s_%.2f.txt' % (attribution_method, method, alpha)
        bb_file = os.path.join(out_path, bb_name)
        (err, res, overlap, no_mask, num_blacklist) = compute_localization_results(
            bb_file=bb_file, 
            imdb_file=imdb_file,  
            annotation_dir=annotation_dir,
            verbose=verbose
        )
        errs[i] = err
        results.append(res)
        overlaps.append(overlap)
        num_blacklists.append(num_blacklist)
        no_masks.append(no_mask)
    for i in range(len(alphas)):
        print('alpha = %.2f, err = %f' % (alphas[i], errs[i]))
    min_i = np.argmin(errs)
    print('best alpha = %.2f, err = %f' % (alphas[min_i], errs[min_i]))
    results = {
        'imdb_file': imdb_file,
        'annotation_dir': annotation_dir,
        'attribution_method': attribution_method,
        'method': method,
        'out_path': out_path,
        'alphas': alphas,
        'errors': errs,
        'example_indicators': results,
        'example_overlaps': overlaps,
        'num_blacklists': num_blacklists,
        'no_masks': no_masks,
        'best_index': min_i,
        'best_alpha': alphas[min_i],
        'best_err': errs[min_i],
    }
    return results


def get_bb_file(out_path,
                attribution_method,
                method,
                alpha,
                smooth=0.,
                processing=None):
    """
    Return path of bounding box file.

    Args:
        out_path: String.
        attribution_method: String.
        method: String.
        alpha: Float.
        smooth: Float.
        processing: String.

    Return:
        out_file: String, path to bounding box file.
    """
    if processing is None:
        if smooth == 0.:
            bb_file = ('bb_val_%s_%s_%.2f.txt'
                       % (attribution_method, method, alpha))
        else:
            bb_file = ('bb_val_%s_%s_%.2f_sm_%.1f.txt'
                       % (attribution_method, method, alpha, smooth))
    else:
        bb_file = ('bb_val_%s_%s_%s_%.2f_sm_%.1f.txt'
                   % (attribution_method, processing, method, alpha, smooth))

    out_file = os.path.join(out_path, bb_file)
    return out_file


def get_bbox_and_localization_results(
    attribution_method='perturbations',
    data_dir='/scratch/shared/slow/vedaldi/vis/exp20-sal-im12val-vgg16',
    out_path='/scratch/shared/slow/mandela/bbox_results',
    method='mean',
    alphas=np.arange(0,10,0.1),
    annotation_dir='/datasets/imagenet14/cls_loc/val',
    imdb_file='./data/val_imdb_0_1000.txt',
    verbose=True,
    smooth=0.,
    processing=None,
    analysis_file=None,
):  
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i in range(len(alphas)):
        alpha = alphas[i]

        # Get name of bounding box file.
        out_file = get_bb_file(out_path=out_path,
                               attribution_method=attribution_method,
                               method=method,
                               alpha=alpha,
                               smooth=smooth,
                               processing=processing)

        # Generate bounding box file if it doesn't exist.
        if not os.path.exists(out_file):
            generate_bbox_file(data_dir=data_dir,
                               out_file=out_file,
                               method=method,
                               alpha=alpha,
                               imdb_file=imdb_file,
                               smooth=smooth,
                               processing=processing,
                               analysis_file=analysis_file)

    res = find_best_alpha(
        imdb_file=imdb_file, 
        annotation_dir=annotation_dir,
        attribution_method=attribution_method,
        method=method,
        alphas=alphas,
        verbose=verbose,
        out_path=out_path
    )

    torch.save(res, os.path.join(out_path, '%s_%s_bbox_dict_new_v2.pth' % (attribution_method, method)))


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', str2bool)
        parser.add_argument('--attribution_method', type=str, default='pertrubations')
        parser.add_argument('--data_dir', type=str, default='/scratch/shared/slow/vedaldi/vis/exp20-sal-im12val-vgg16')
        parser.add_argument('--out_path', type=str, default='/scratch/shared/slow/mandela/bbox_results_smooth_20')
        # parser.add_argument('--out_path', type=str, default='/scratch/shared/slow/ruthfong/imagenet_localization/bbox_results')
        parser.add_argument('--method', type=str, default='mean')
        parser.add_argument('--annotation_dir', type=str, default='/datasets/imagenet14/cls_loc/val')
        parser.add_argument('--imdb_file', type=str, default='./data/val_imdb_0_1000.txt')
        parser.add_argument('--verbose', type='bool', default=True)
        parser.add_argument('--gpu', type=int, nargs='*', default=None)
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

        args = parser.parse_args()

        if args.method == "mean":
            the_range = np.arange(0,10,0.5)
        else:
            the_range = np.arange(0,1,0.05)

        get_bbox_and_localization_results(
            attribution_method=args.attribution_method,
            data_dir=args.data_dir,
            out_path=args.out_path,
            method=args.method,
            alphas=the_range,
            annotation_dir=args.annotation_dir,
            imdb_file=args.imdb_file,
            verbose=args.verbose,
            smooth=args.smooth,
            processing=args.processing,
            analysis_file=args.analysis_file,
        )
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
