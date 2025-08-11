import zarr
import os
import numpy as np
import argparse
import tifffile as tiff

def extract(path):
    with tiff.TiffFile(path) as tif:
        pages = tif.pages
        z_planes = [page.asarray() for page in pages]
    return np.stack(z_planes, axis=0)

def main(args):
    in_path = args.in_path
    out_path = args.out_path

    images = [] # list of frame paths

    for path in sorted(os.listdir(in_path)):
        if path.endswith(".tif"):
            full_path = os.path.join(in_path, path)
            images.append(full_path)
    print(images)
    for i, img_path in enumerate(images):
        image = extract(img_path)
        out_file_path = os.path.join(out_path, f"t{i:03d}")
        z = zarr.open(out_file_path, mode='w', shape=image.shape, chunks=(100, 100, 100), dtype=image.dtype)
        z[:] = image
        print(f"Time frame {i} of {len(images)} processed!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run tiff to zarr script for training.")
    parser.add_argument("--in_path", type=str, required=True)    
    parser.add_argument("--out_path", type=str, required=True)    
    args = parser.parse_args()
    
    main(args)