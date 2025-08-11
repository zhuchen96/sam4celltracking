#!/bin/bash
set -e 
# deactivate virtualenv if already active 
if command -v conda deactivate > /dev/null; then conda deactivate; fi 

if conda info --envs | grep -q "^sam4celltracking "; then 
    echo "Virtual environment 'sam4celltracking' already exists, skipping creation."
    conda activate sam4celltracking
else
    echo "Creating virtual environment 'sam4celltracking'..."
    conda create -n sam4celltracking python=3.10  
    # Activate the new environment
    conda activate sam4celltracking
    echo "Installing dependencies..."
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    else
        echo "requirements.txt not found, skipping dependency installation."
    fi
fi
echo "Environment activated: $CONDA_PREFIX"

echo "Start tracking..."
python tracking_3d_general.py \
    --dataset "TRIF" \
    --image_path "../Data/Fluo-N3DL-TRIF/01_zarr" \
    --detection_path "../Data/Fluo-N3DL-TRIF/01_detection" \
    --res_path "../Data/Fluo-N3DL-TRIF/01_RES" \
    --window_size 64 \
    --init_json_path "../Data/Fluo-N3DL-TRIF/01_centroid.json" \
    --neighbor_thresh 20 \
    --model "../trained_models/sam_3d_trif.pth" \
    --z_divisor 1 \
    --top_thresh 0.8 \
    --diff_thresh 0.1 \
    --min_voxels_general 1000 \
    --min_voxels_no_neighbor 1000 \
    --edge_guard_xy 5 \
    --edge_guard_z_low 2 \
    --edge_guard_z_high 2

set +e

