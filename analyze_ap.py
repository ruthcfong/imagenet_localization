import torch
import numpy as np

def analyze(ap_file, method='perturbations'):
    ap_dict = torch.load(ap_file)


    total_ap = 0
    total_images = 0
    class_mean_ap_dict = {}
    for key in ap_dict.keys():
        class_ap_dict = ap_dict[key]['aps']
        class_mean_ap_dict[key] = np.mean(class_ap_dict)
        total_ap += np.sum(class_ap_dict)
        total_images += len(class_ap_dict)
    torch.save(class_mean_ap_dict, './%s_class_mean_ap_dict.pth' % method)
    average_ap = total_ap/float(total_images)
    print("Average AP for %s: %.3f" % (method, average_ap))
    torch.save(average_ap, './%s_avg_ap.pth' % method)


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--ap_file', type=str, default='./ap.pth')
        parser.add_argument('--method', type=str, default='perturbations')
        args = parser.parse_args()

        analyze(args.ap_file, args.method)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)