# Segment Anything for Cell Tracking

Official implementation of our **MedAGI Workshop (MICCAI 2025)** paper. Arxiv: [https://arxiv.org/abs/2509.09943](https://arxiv.org/abs/2509.09943) 

---

## Overview
We introduce a **generalized cell linking and tracking framework** built on **Segment Anything 2 (SAM2)** [1].  
Our method is designed for **cell segmentation and lineage tracking** in microscopy videos and large-scale volumetric datasets.  

### Key Contributions
- **Unified linking** across 2D and 3D datasets  
- **Mask propagation & reconstruction** to recover missing cells across time frames  
- **Mitosis detection** based on mask splitting and lineage tracking  
- **Zero-shot 2D mode** (no training required)  
- **Fine-tuned 3D mode** using **SAM-Med3D**[2] (trained with automatically-generated masks)
- **Scalable** pipelines for large-scale datasets using ZARR  


---

## Features
- ✅ Zero-shot **mask linking** across time frames using pretrained SAM2  
- ✅ **Fine-tuning pipeline** for SAM-Med3D with coarse annotations  
- ✅ **Joint segmentation and tracking** for large 3D+t datasets  
- ✅ Support for **Cell Tracking Challenge (CTC) format** inputs and outputs  
- ✅ Modular preprocessing: TIFF ↔ ZARR conversion, patch extraction, mask initialization  
- ✅ Dataset-specific bash scripts for reproducible experiments  

---

## Data Folder Organization

### 1. For 2D / 3D CTC Datasets (TIFF format)
```
repo/
│
├── Data/          # all datasets should be here
│   ├── Dataset_name/
│   │   ├── 01/  # raw images
│   │   │   ├── t000.tif
│   │   │   ├── t001.tif
│   │   │   └── ...
│   │   ├── 01_ERR_SEG/. # coarse segmentation results
│   │   │   ├── mask000.tif
│   │   │   ├── mask001.tif
│   │   │   └── ...
│   │   ├── 01_RES/. # output folder (auto-generated)
│   │   │
│   │   ├── 02/
│   │   │   ├── t000.tif
│   │   │   ├── t001.tif
│   │   │   └── ...
│   │   ├── 02_ERR_SEG/
│   │   │   ├── mask000.tif
│   │   │   ├── mask001.tif
│   │   │   └── ...
│   │   ├── 02_RES/. 
│   └── ... more sequences ...
│
└── src/           # code folder
```

### 2. For Large-Scale 3D+t Datasets (ZARR format)
```
repo/
│
├── Data/          # all datasets should be here
│   ├── Dataset_name/
│   │   ├── 01/  # raw images (in ZARR format)
│   │   │   ├── t000
│   │   │   ├── t001
│   │   │   └── ...
│   │   ├── 01_detection/ (detected cell centers)
│   │   │   ├── d000.tif
│   │   │   ├── d001.tif
│   │   │   └── ...
│   │   ├── 01_centroid.json (Tracks initialized with the GUI)
│   │   ├── 01_RES/. # output folder (auto-generated)
│   │   │
│   │   ├── 02/ 
│   │   │   ├── t000
│   │   │   ├── t001
│   │   │   └── ...
│   │   ├── 02_detection/
│   │   │   ├── d000.tif
│   │   │   ├── d001.tif
│   │   │   └── ...
│   │   ├── 02_centroid.json
│   │   ├── 02_RES/.
│   └── ... more sequences ...
│
└── src/           # code folder
```


⚠️ **Note:** Filenames must follow **zero-padded indices** (`t000.tif`, `mask000.tif`) for compatibility.  

---

## Installation

### Requirements
- Linux  
- Conda  
- Python 3.10  
- CUDA-enabled GPU (tested on NVIDIA RTX 4090)  

### Setup
```bash
git clone https://github.com/zhuchen96/sam4celltracking.git
cd sam4celltracking/src
bash -i prepare_software.sh   # create environment
conda activate sam4celltracking
```

### **Steps to Run Linking Scripts**  
1. Download the pretrained SAM2 model from [**Download**](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) and save it under `src/trained_models`
2. Identify the bash script for each dataset in `linking_scripts` folder, named in the format `DatasetName-SequenceID.sh`.  
3. Run the bash script using the following command:  
   ```bash
   bash -i linking_scripts/DatasetName-SequenceID.sh
   ```

### **Steps to Run Tracking Scripts**  
1. Download the pretrained SAM2 model from [**Download**](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) and save it under `src/trained_models`
2. Download the pretrained model for specific dataset from [**Download**](xxx) and save it under `src/trained_models`
3. Convert TIFF files to ZARR files:
    ```bash
    python preprocessing/tiff_to_zarr.py --in_path path_to_tiff_images --out_path path_to_save_zarr_files
    ```
4. Initialize tracks in the first time frame
    ```bash
    python preprocessing/initialize_tracks.py --in_mask_path initial_detections_in_first_time_frame(TIFF) --out_path path_to_save_output_json_file
    ```  
5. Identify the bash script for each dataset in `tracking_scripts` folder, named in the format `DatasetName-SequenceID.sh`.  
6. Run the bash script using the following command:  
   ```bash
   bash -i tracking_scripts/DatasetName-SequenceID.sh
   ```

### **Steps to Finetune SAM-Med3D**  
1. Download the pretrained SAM-Med3D model from [**Download**](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view?usp=sharing) and save it under `src/CTC_submission/trained_models`
2. Prepare image patches for training \
   For normal 3D datasets:
   ```bash
   python preprocessing/training_data_processing.py --dataset your_dataset_name(e.g.Fluo-N3DH-CHO-01) --img_path path_to_raw_images --mask_path path_to_mask_files
   ```
   For large-scale 3D datasets:
   - Convert TIFF image files to ZARR files:
        ```bash
        python preprocessing/tiff_to_zarr.py --in_path path_to_tiff_images --out_path path_to_zarr_files
        ```
   - Extract image patches from ZARR files
        ```bash
        python preprocessing/training_data_processing_zarr.py --dataset your_dataset_name(e.g.Fluo-N3DL-TRIC-01) --img_path path_to_zarr_files --mask_path path_to_mask_tiff_files
        ```   

3. Modify the path to image patches for training in `src/sam_med3d/utils/data_paths.py`. It is also possible to list several paths to train the network with more datasets.
4. Run the training script
   ```bash
   python train.py --task_name your_task_name
   ```

## **Output**
The algorithm generates the following outputs in the specified result folder:  
1. TIFF files named `mask{i}.tif`, where `i` represents the time frame index.  
2. A tracking information file in cell tracking challenge format, `res_track.txt`.  

---


## Contact
For questions or collaborations, please contact:  
📧 zhu.chen@lfb.rwth-aachen.de 

---

## **Citations**
- [1] Zhu, J., Qi, Y., & Wu, J. (2024). Medical SAM 2: Segment medical images as video via Segment Anything Model 2. arXiv preprint arXiv:2408.00874.
- [2] Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z., Shen, Y., Fu, B., Zhang, S., He, J., & Qiao, Y. (2023). SAM-Med3D. arXiv preprint arXiv:2310.15161.

