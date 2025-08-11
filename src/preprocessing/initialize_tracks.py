import os
import json
import argparse
import numpy as np
import tifffile as tiff
from scipy.ndimage import label, center_of_mass

def get_init_centroid(init_mask_path):
    init_mask_arr = tiff.imread(init_mask_path)
    init_mask_values = np.unique(init_mask_arr)
    
    centroids = {}

    for mask_value in init_mask_values:
        if mask_value == 0:  # Skip background
            continue
        mask_coords = np.where(init_mask_arr == mask_value)
        centroid = np.mean(mask_coords, axis=1)
        centroids[int(mask_value)] = centroid.tolist()

    return centroids

def process_tiff_file(file_path):
    print("loading")
    tiff_data = tiff.imread(file_path)
    
    zero_mask = (tiff_data == 0)
    print("done")
    labeled_data, num_features = label(zero_mask, structure=np.ones((3, 3, 3)))
    print("finding center")
    centroids = center_of_mass(zero_mask, labeled_data, range(1, num_features + 1))
    print("Done")
    centroids = [tuple(map(float, centroid)) for centroid in centroids]
    return centroids

def process_tiff_files(input_dir, output_file):
    centroids = get_init_centroid(input_dir)
    
    with open(output_file, 'w') as json_file:
        json.dump(centroids, json_file, indent=4)
    
    print(f"Centroids have been saved to {output_file}")

def main(args):
    in_mask_path = args.in_mask_path
    out_json = args.out_json

    process_tiff_files(in_mask_path, out_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tiff to zarr script for training.")
    parser.add_argument("--in_mask_path", type=str, required=True)    
    parser.add_argument("--out_json", type=str, required=True)    
    args = parser.parse_args()
    
    main(args)