import os
import argparse
import numpy as np
import tifffile as tiff

def normalization(img_arr):
    perc_min = np.percentile(img_arr, 1)
    perc_max = np.percentile(img_arr, 99.8)
    
    img_arr -= int(perc_min)
    diff = perc_max - perc_min
    img_arr /= diff
    img_arr = np.clip(img_arr, 1e-5, 1)
    return img_arr

def main(args):
    dataset = args.dataset

    images_path = [args.img_path]
    masks_path = [args.mask_path]
    images = [] # list of image paths
    masks = []  # list of mask paths

    output_path_img = f"../Data/Training_data/{dataset}/imagesTr"
    output_path_mask = f"../Data/Training_data/{dataset}/labelsTr"
    os.makedirs(output_path_img, exist_ok=True)
    os.makedirs(output_path_mask, exist_ok=True)
    
    crop_size = (64, 128, 128)

    for img_folder, mask_folder in zip(images_path, masks_path):

        for path in sorted(os.listdir(img_folder)):
            if path.endswith(".tif"):
                full_path = os.path.join(img_folder, path)
                images.append(full_path)

        for path in sorted(os.listdir(mask_folder)):
            if path.endswith(".tif"):
                full_path = os.path.join(mask_folder, path)
                masks.append(full_path)

    for img in range(0, len(images), 10):
        image_arr = tiff.imread(images[img])
        mask_arr = tiff.imread(masks[img])

        image_arr = image_arr.astype(np.float32)
        mask_arr = mask_arr.astype(np.uint8)

        image_arr = normalization(image_arr) 

        for i in range(30): 
            ind = (img // 10) * 30 + i
            mean_value = 0

            while mean_value <= 0.2: 
                
                rnd_start = [np.random.randint(0, max(1,mask_dim-patch_dim)) for patch_dim, mask_dim in zip(crop_size, mask_arr.shape)]
                rnd_end = [start+patch_dim for start, patch_dim in zip(rnd_start, crop_size)]
                slicing = tuple(map(slice, rnd_start, rnd_end))

                img_tmp = image_arr[slicing]
                mask_tmp = mask_arr[slicing]

                mean_value = mask_tmp.mean()

            tiff.imwrite(os.path.join(output_path_img, f't={ind:03d}.tif'), img_tmp)
            tiff.imwrite(os.path.join(output_path_mask, f't={ind:03d}.tif'), mask_tmp)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run preprocessing script for training.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--img_path", type=str, required=True)    
    parser.add_argument("--mask_path", type=str, required=True)    
    parser.add_argument("--seed", type=int, default=2023, help="Seed")  
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    main(args)