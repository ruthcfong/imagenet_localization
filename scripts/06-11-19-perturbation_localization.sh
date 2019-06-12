#!/usr/bin/env bash
#exp_name="exp10-imnet-vgg16-variant2"
exp_name="exp11-imnet-vgg16-variant2-init"
data_dir="/checkpoint/vedaldi/saliency/${exp_name}"
analysis_file="/checkpoint/ruthfong/attribution/results/${exp_name}.pth"
image_dir="/checkpoint/ruthfong/attribution/results/${exp_name}.pth"
annotation_dir="/private/home/ruthfong/data/imagenet14/cls_loc/val"
out_path="/checkpoint/ruthfong/attribution/bbox_results"

methods=( "mean" "threshold" "energy" )
processings=( "mean_crossover" "single_crossover" )
smooths=( 0 10 20 )

for method in "${methods[@]}"; do
    for processing in "${processings[@]}"; do
        for smooth in "${smooths[@]}"; do
            python bbox_and_localization.py \
                --data_dir $data_dir \
                --image_dir $image_dir \
                --out_path $out_path \
                --method $method \
                --annotation_dir $annotation_dir \
                --smooth $smooth \
                --processing $processing \
                --analysis_file $analysis_file \
                --exp_name $exp_name
        done
    done
done