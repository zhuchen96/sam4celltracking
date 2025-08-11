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

# Run the linking algorithm with eight input parameters:
# input_sequence mask_sequence output_sequence 2d_or_3d window_size dis_threshold neighbor_dist pretrained_model-required_by_3d_mask_generation_mode 
./linking.sh "../Data/DIC-C2DH-HeLa/01" "../Data/DIC-C2DH-HeLa/01_ERR_SEG" "../Data/DIC-C2DH-HeLa/01_RES" "2d" 512 1200 20 ""

set +e


