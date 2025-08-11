# **Segment Anything for Cell Tracking**

## **Overview**  
- This repository contains a generalized cell-linking / cell-tracking algorithm based on **SegmentAnything2 (SAM2)** [1]. 
- The linking algortihm supports both 2D and 3D datasets. It can link masks that belong to the same cell across time frames, generate new masks for cells missing in intermediate time frames and detect cell mitosis. 
  - For 2D datasets, the linking algorithm operates as a zero-shot method, leveraging a pretrained SAM2 model without requiring any additional training or fine-tuning.
  - For 3D datasets, the algorithm offers two modes:
    - Linking-only Mode: Suitable for datasets with a nearly complete set of generated masks. This mode is training-free and computationally efficient.
    - Mask-generation Mode: Designed for datasets with numerous missing masks. This mode utilizes a fine-tuned **SAM-Med3D** [2] model to generate missing 3D masks for intermediate time frames.
- The tracking algorithm is designed for large-scale 3D datasets. The **SAM-Med3D** [2] model can be fine-tuned with coarse segmentation masks. And the tracking and segmentation is done jointly using pretrained SAM2 and fine-tuned SAM-Med3D.

---

## **Usage Instructions**

### **Prerequisites**
- System: Linux
- Conda  
- Python 3.10

### **Steps to Run Linking Scripts**  
1. Clone the repository
   ```bash
   git clone https://github.com/zhuchen96/sam4celltracking.git
   ```
2. Go to the code folder
   ```bash
   cd src
   ```
3. Download the pretrained SAM2 model from [**Download**](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) and save it under `src/trained_models`
4. Identify the bash script for each dataset in `linking_scripts` folder, named in the format `DatasetName-SequenceID.sh`.  
5. Run the bash script using the following command:  
   ```bash
   bash -i linking_scripts/DatasetName-SequenceID.sh
   ```

### **Steps to Run Tracking Scripts**  
1. Clone the repository
   ```bash
   git clone https://github.com/zhuchen96/sam4celltracking.git
   ```
2. Go to the code folder
   ```bash
   cd src
   ```
3. Download the pretrained SAM2 model from [**Download**](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) and save it under `src/trained_models`
4. Download the pretrained model for specific dataset from [**Download**](xxx) and save it under `src/trained_models`
5. Convert TIFF files to ZARR files:
    ```bash
    python preprocessing/tiff_to_zarr.py --in_path path_to_tiff_images --out_path path_to_save_zarr_files
    ```
6. Initialize tracks in the first time frame
    ```bash
    python preprocessing/initialize_tracks.py --in_mask_path initial_detections_in_first_time_frame(TIFF) --out_path path_to_save_output_json_file
    ```  
7. Identify the bash script for each dataset in `tracking_scripts` folder, named in the format `DatasetName-SequenceID.sh`.  
8. Run the bash script using the following command:  
   ```bash
   bash -i tracking_scripts/DatasetName-SequenceID.sh
   ```

### **Steps to Finetune SAM-Med3D**  
1. Clone the repository
   ```bash
    git clone https://github.com/zhuchen96/sam4celltracking.git
   ```
2. Go to the code folder
   ```bash
   cd src
   ```
3. Run the following bash file to generate a new conda environment
   ```bash
   bash -i prepare_software.sh
   ```  
4. Activate the conda environment
5. Download the pretrained SAM-Med3D model from [**Download**](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view?usp=sharing) and save it under `src/CTC_submission/trained_models`
6. Prepare image patches for training \
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

7. Modify the path to image patches for training in `src/sam_med3d/utils/data_paths.py`. It is also possible to list several paths to train the network with more datasets.
8. Run the training script
   ```bash
   python train.py --task_name your_task_name
   ```

## **Output**
The algorithm generates the following outputs in the specified result folder:  
1. TIFF files named `mask{i}.tif`, where `i` represents the time frame index.  
2. A tracking information file in cell tracking challenge format, `res_track.txt`.  


## **Citations**
- [1] Zhu, J., Qi, Y., & Wu, J. (2024). Medical SAM 2: Segment medical images as video via Segment Anything Model 2. arXiv preprint arXiv:2408.00874.
- [2] Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z., Shen, Y., Fu, B., Zhang, S., He, J., & Qiao, Y. (2023). SAM-Med3D. arXiv preprint arXiv:2310.15161.