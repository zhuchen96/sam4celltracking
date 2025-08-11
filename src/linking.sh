#!/bin/bash
echo "Running linking script for: $1"
if [[ "$4" == "2d" ]]; then
    echo "Processing 2D data..."
    python linking_2d_general.py --image_path $1 --mask_path $2 --out_path $3 --size $5 --dis_threshold $6 --neighbor_dist $7
elif [[ "$4" == "3d" && -n "$8" ]]; then
    echo "Processing 3D data with mask generation..."
    python linking_3d_general_gen_mask.py --image_path $1 --mask_path $2 --out_path $3 --size $5 --dis_threshold $6 --model $8
elif [[ "$4" == "3d"  && -z "$8" ]]; then
    echo "Processing 3D data..."
    python linking_3d_general_linking_only.py --image_path $1 --mask_path $2 --out_path $3 --size $5 --dis_threshold $6 --neighbor_dist $7
else
    echo "Error: fourth argument must be '2d' or '3d'."
    exit 1
fi

echo "Image path: $1"
echo "Mask path: $2"
echo "Output path: $3"
echo "Data type: $4"
echo "Window size: $5"
echo "Disappear threshold: $6"
echo "Neighbor threshold: $7"
echo "Pretrained model: $8"
