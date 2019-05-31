#!/usr/bin/env bash
python3 bbox_and_localization.py
python3 bbox_and_localization.py --attribution_method guided_backprop --data_dir /scratch/shared/slow/mandela/guided_backprop
python3 bbox_and_localization.py --attribution_method gradient --data_dir /scratch/shared/slow/mandela/gradient

python3 bbox_and_localization.py --method min_max_diff --small_range True
python3 bbox_and_localization.py --attribution_method guided_backprop --data_dir /scratch/shared/slow/mandela/guided_backprop --method min_max_diff --small_range True
python3 bbox_and_localization.py --attribution_method gradient --data_dir /scratch/shared/slow/mandela/gradient --method min_max_diff --small_range True

python3 bbox_and_localization.py --method energy --small_range True
python3 bbox_and_localization.py --attribution_method guided_backprop --data_dir /scratch/shared/slow/mandela/guided_backprop --method energy --small_range True
python3 bbox_and_localization.py --attribution_method gradient --data_dir /scratch/shared/slow/mandela/gradient --method energy --small_range True