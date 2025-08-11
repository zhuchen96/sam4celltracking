#!/bin/bash
echo "Running tracking script for: $1"
echo "Start tracking..."
python tracking_3d_general.py -- dataset "TRIC" ../Data/Fluo-N3DL-TRIC/01_zarr --image_path $1 --mask_path $2 --out_path $3 --size $5 --dis_threshold $6 --neighbor_dist $7

echo "Image path: $1"
echo "Mask path: $2"
echo "Output path: $3"
echo "Data type: $4"
echo "Window size: $5"
echo "Disappear threshold: $6"
echo "Neighbor threshold: $7"
echo "Pretrained model: $8"
