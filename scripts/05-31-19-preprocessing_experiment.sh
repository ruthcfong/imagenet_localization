#!/usr/bin/env bash

#smooths=( "0.0" "10.0" "20.0" )
#methods=( "mean" "min_max_diff" "energy" )
smooths=( "10.0" "20.0" )
methods=( "min_max_diff" )
processings=( "mean_crossover" "single_crossover" )

for smooth in "${smooths[@]}"; do
    for method in "${methods[@]}"; do
        for processing in "${processings[@]}"; do
            log_file="./logs/bbox_loc_log_${method}_${processing}_${smooth}.out"
            echo "Launching job logging to ${log_file}"
            python3 bbox_and_localization.py \
                --method ${method} \
                --processing ${processing} \
                --smooth ${smooth} &> ${log_file} &
        done
    done
done

