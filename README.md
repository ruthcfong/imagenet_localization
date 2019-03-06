# imagenet_localization

This repo contains Python code for generating bounding box files which can be used to compute ImageNet localization results.

## 0. Prerequisites.
Run `pip install -r requirements.txt` to install the needed packages.

## 1. Generate bounding box file with predicted labels and bounding boxes.

Run `python generate_bbox_file.py DATA_DIR OUT_FILE`, where `DATA_DIR` is the path to the directory containing torch result files and `OUT_FILE` is the path for the output file.

## 2. Compute localization results.

Run `python compute_localization_results.py --bb_file BB_FILE --annotation_di ANNOTATION_DIR`, where `BB_FILE` is the path to the output file from the first steph and `ANNOTATION_DIR` is the path to the directory containing ImageNet XML annotation files.
