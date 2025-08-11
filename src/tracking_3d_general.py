import os
import csv
import json
import zarr
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tifffile as tiff
import torch
from scipy.spatial.distance import cosine

from utils.segmentation1 import Segmenter
from sam2.apply_sam2 import build_sam2
from utils.visualize import show_mask, show_points, show_box
from utils.tools import (
    find_closest_nonzero_and_keep_region_3d,
    get_prompt_bb_3d,
    extract_zarr_3d,
)

def check_overlap(mask, save_indices, mask_bb):
    queried_values = mask[tuple(save_indices.T)]
    non_zero_count = np.count_nonzero(queried_values)
    total_count = queried_values.size
    percent_non_zero = (non_zero_count / total_count) * 100
    return percent_non_zero

def get_init_centroid(init_mask_path):
    init_mask_arr = tiff.imread(init_mask_path)
    init_mask_values = np.unique(init_mask_arr)
    centroids = {}
    for mask_value in init_mask_values:
        if mask_value == 0:  # Skip background
            continue
        mask_coords = np.where(init_mask_arr == mask_value)
        centroid = np.mean(mask_coords, axis=1)
        centroids[mask_value] = centroid
    return centroids

def get_centroid_twang(twang_path, tf, z_divisor=1):
    centroids = []
    seed_path = os.path.join(twang_path, f"t{tf:03d}_item_0005_ExtractSeedBasedIntensityWindowFilter_.csv")
    with open(seed_path, mode='r') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            zpos = int(row['zpos'])
            if z_divisor != 1:
                zpos = int(zpos // z_divisor)
            centroids.append([zpos, int(row['ypos']), int(row['xpos'])])
    return centroids

def process_lists(list1, list2, top_thresh, diff_thresh):
    if len(list2) == 1:
        return list1, list2
    indices = sorted(range(len(list2)), key=lambda i: list2[i], reverse=True)[:2]
    highest_values = [list2[i] for i in indices]
    filtered_list1 = [list1[i] for i in indices]
    filtered_list2 = highest_values
    if highest_values[0] < top_thresh and (highest_values[0] - highest_values[1]) < diff_thresh:
        return filtered_list1, filtered_list2
    else:
        return [filtered_list1[0]], [filtered_list2[0]]

def filter_coordinates_within_bb(coordinates, bounding_box):
    coordinates = np.array(coordinates)
    z_min, z_max, y_min, y_max, x_min, x_max = bounding_box
    mask = (
        (coordinates[:, 0] >= z_min) & (coordinates[:, 0] <= z_max) &
        (coordinates[:, 1] >= y_min) & (coordinates[:, 1] <= y_max) &
        (coordinates[:, 2] >= x_min) & (coordinates[:, 2] <= x_max)
    )
    return coordinates[mask]

def filter_twang_coordinates(coordinate_list, center, threshold):
    coordinates = np.array(coordinate_list)
    center_array = np.array(center)
    distances = np.linalg.norm(coordinates - center_array, axis=1)
    return coordinates[distances < threshold]

def get_shift_generic(predictor, pos, bb_points, img_list, tf, mask_ind,
                      size, image_y, image_x, out_path, visualize):
    overlay_size = size
    half_overlay_size = overlay_size // 2

    x = pos[2]; y = pos[1]; z = pos[0]

    start_x = x - half_overlay_size
    end_x = start_x + overlay_size
    start_y = y - half_overlay_size
    end_y = start_y + overlay_size

    patches = [extract_zarr_3d(p, z, start_y, end_y, start_x, end_x, overlay_size, image_y, image_x) for p in img_list]

    img_list_imgs = [Image.fromarray(img.astype('uint8')) for img in patches]

    bb_local = bb_points - [start_y, start_x, start_y, start_x]

    ann_frame_idx = 0
    ann_obj_id = 1

    inference_state = predictor.init_state_2(img_list=img_list_imgs)
    predictor.reset_state(inference_state)

    box = np.array(bb_local, dtype=np.float32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )

    if visualize:
        plt.figure(figsize=(12, 8))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.fromarray(patches[ann_frame_idx]))
        show_box(box, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        mask_folder_name = os.path.join(out_path, f"vis/masks{mask_ind:04d}")
        os.makedirs(mask_folder_name, exist_ok=True)
        plt.savefig(os.path.join(mask_folder_name, f'mask_{tf:04d}_p.png'), dpi=300, bbox_inches='tight')
        plt.close()

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    out_mask = video_segments[1][1]

    if visualize:
        plt.figure(figsize=(12, 8))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.fromarray(patches[-1]))
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        mask_folder_name = os.path.join(out_path, f"vis/masks{mask_ind:04d}")
        os.makedirs(mask_folder_name, exist_ok=True)
        plt.savefig(os.path.join(mask_folder_name, f'mask_{tf:04d}_p2.png'), dpi=300, bbox_inches='tight')
        plt.close()

    out_mask = find_closest_nonzero_and_keep_region_3d(out_mask, (int(size/2), int(size/2)))
    true_indices = np.argwhere(out_mask)
    return np.mean(true_indices, axis=0)

def compare_features_generic(predictor, pos_old, pos_new_list, bb_points, img_list, tf, mask_ind,
                             size, image_y, image_x, out_path, visualize):
    overlay_size = size
    half_overlay_size = overlay_size // 2

    x = pos_old[2]; y = pos_old[1]; z = pos_old[0]

    start_x = x - half_overlay_size
    end_x = start_x + overlay_size
    start_y = y - half_overlay_size
    end_y = start_y + overlay_size

    patch = extract_zarr_3d(img_list[0], z, start_y, end_y, start_x, end_x, overlay_size, image_y, image_x)
    patch_img = Image.fromarray(patch.astype('uint8'))

    bb_local = bb_points - [start_y, start_x, start_y, start_x]

    ann_frame_idx = 0
    ann_obj_id = 1

    inference_state = predictor.init_state_2(img_list=[patch_img])
    predictor.reset_state(inference_state)

    box = np.array(bb_local, dtype=np.float32)
    features = predictor.add_new_points_or_box_2(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )
    mask_old_features = features["maskmem_features"][0].to(torch.float32).cpu().numpy()

    if visualize:
        plt.figure(figsize=(12, 8))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(patch_img)
        show_box(box, plt.gca())
        mask_folder_name = os.path.join(out_path, f"vis/masks{mask_ind:04d}")
        os.makedirs(mask_folder_name, exist_ok=True)
        plt.savefig(os.path.join(mask_folder_name, f'mask_{tf:04d}_p.png'), dpi=300, bbox_inches='tight')
        plt.close()

    mask_new_features_list = []
    for pos_new_id, pos_new in enumerate(pos_new_list):
        x = pos_new[2]; y = pos_new[1]; z = pos_new[0]

        start_x = x - half_overlay_size
        end_x = start_x + overlay_size
        start_y = y - half_overlay_size
        end_y = start_y + overlay_size

        patch = extract_zarr_3d(img_list[1], z, start_y, end_y, start_x, end_x, overlay_size, image_y, image_x)
        patch_img2 = Image.fromarray(patch.astype('uint8'))

        inference_state = predictor.init_state_2(img_list=[patch_img2])
        predictor.reset_state(inference_state)

        box = np.array(bb_local, dtype=np.float32)
        features = predictor.add_new_points_or_box_2(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )
        mask_new_features_list.append(features["maskmem_features"][0].to(torch.float32).cpu().numpy())

    similarity = []
    mask_old_features = mask_old_features.astype(np.float32).flatten()
    for new_feature in mask_new_features_list:
        new_feature = new_feature.astype(np.float32).flatten()
        similarity.append(1 - cosine(mask_old_features, new_feature))

    return similarity

# ----------------------- Main pipeline (dataset-routed) -----------------------

def main(args):
    dataset = args.dataset
    image_path = args.image_path
    detection_path = args.detection_path
    res_path = args.res_path
    window_size = args.window_size
    init_json_path = args.init_json_path
    neighbor_thresh = args.neighbor_thresh
    z_divisor = args.z_divisor
    top_thresh = args.top_thresh
    diff_thresh = args.diff_thresh
    min_voxels_no_neighbor = args.min_voxels_no_neighbor
    edge_guard_xy = args.edge_guard_xy
    edge_guard_z_low = args.edge_guard_z_low
    edge_guard_z_high = args.edge_guard_z_high
    min_voxels_general = args.min_voxels_general
    visualize = args.visualize
    seed = args.seed
    model = args.model

    # Build input lists
    images = sorted([os.path.join(image_path, p) for p in os.listdir(image_path)])

    store = zarr.open(image_path[0], mode="r")
    image_shape = store.shape
    print(f"Loaded Zarr file shape: {image_shape}")
    image_z, image_y, image_x = image_shape

    start_tf = 0
    end_tf = len(images)

    # Build SAM3D and SAM2
    segmenter = Segmenter(-1, image_path, image_z, image_y, image_x, model)
    sam2 = build_sam2()

    temp_dict = {}
    output_dict = {}
    mask_bb = [0, image_z - 1, 0, image_y - 1, 0, image_x - 1]
    new_ind = 0
    new_born_ids = []

    for tf in range(start_tf, end_tf):
        mask = np.zeros((image_z, image_y, image_x))
        print(f"Processing time frame {tf:03d}")

        segmenter.update_image(tf)

        if tf == start_tf:
            with open(init_json_path) as json_file:
                positions = json.load(json_file)

            for current_ind, position in positions.items():
                current_ind = int(current_ind)
                if current_ind % 50 == 0:
                    print(f"Tracking cell: {current_ind:03d}")

                x = int(position[2]); y = int(position[1]); z = int(position[0])

                new_array = segmenter.segmentation([z, y, x])
                indices = np.argwhere(new_array == 1)

                if indices.shape[0] <= min_voxels_general:
                    print(f"Tracking of cell {current_ind} is stopped. Mask is too small!")
                    continue

                save_indices = indices - [64, 64, 64] + [z, y, x]
                save_indices = filter_coordinates_within_bb(save_indices, mask_bb)

                center_of_mass = np.round(np.mean(save_indices, axis=0)).astype(int)
                x = center_of_mass[2]; y = center_of_mass[1]; z = center_of_mass[0]

                if dataset == "TRIC":
                    if x <= edge_guard_xy or y <= edge_guard_xy:
                        print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                        continue
                    if y >= image_y - edge_guard_xy or x >= image_x - edge_guard_xy:
                        print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                        continue
                else:  # TRIF
                    if x <= edge_guard_xy or y <= edge_guard_xy or z <= edge_guard_z_low:
                        print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                        continue
                    if y >= image_y - edge_guard_xy or x >= image_x - edge_guard_xy or z >= image_z - edge_guard_z_high:
                        print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                        continue

                if check_overlap(mask, save_indices, mask_bb) > 70:
                    print(f"Tracking of cell {current_ind} is stopped. Overlap!")
                    continue
                else:
                    queried_values = mask[tuple(save_indices.T)]
                    zero_mask = (queried_values == 0)
                    save_indices = save_indices[zero_mask]

                bb_points = get_prompt_bb_3d(save_indices, [z, y, x])
                bb_points = np.array(bb_points) + np.array([y, x, y, x])

                temp_dict[current_ind] = {
                    'centroid': [z, y, x],
                    'centroid_old': [z, y, x],
                    'bb_points': bb_points
                }

                mask[tuple(save_indices.T)] = current_ind

                output_dict[current_ind] = {'start': start_tf, 'end': start_tf, 'parent': 0}

                if current_ind >= new_ind:
                    new_ind = current_ind + 1

        else:
            del_id = []
            twang_positions = get_centroid_twang(detection_path, tf, z_divisor=z_divisor)

            for current_ind in list(temp_dict.keys()):
                mitosis = False
                if current_ind % 50 == 0:
                    print(f"Tracking cell: {current_ind:03d}")

                z, y, x = temp_dict[current_ind]['centroid']
                bb_points = temp_dict[current_ind]['bb_points']

                x_old, y_old, z_old = x, y, z

                neighbor_positions = filter_twang_coordinates(twang_positions, [z, y, x], neighbor_thresh)

                if len(neighbor_positions) == 0:
                    print(f"Cell {current_ind}: no neighbor found")
                    shift_vec = get_shift_generic(
                        sam2, [z, y, x], bb_points, images[tf-1:tf+1], tf, current_ind,
                        window_size, image_y, image_x, res_path, visualize
                    )

                    if np.isnan(shift_vec).any():
                        print(f"Cell {current_ind}: not tracked")
                        del_id.append(current_ind)
                        continue

                    x = x + int(shift_vec[2] - window_size/2)
                    y = y + int(shift_vec[1] - window_size/2)
                    # z unchanged

                    new_array = segmenter.segmentation([z, y, x])
                    indices = np.argwhere(new_array == 1)

                    if indices.shape[0] <= min_voxels_no_neighbor and not mitosis:
                        del_id.append(current_ind)
                        print(f"Tracking of cell {current_ind} is stopped. Mask is too small!")
                        continue

                    save_indices = indices - [64, 64, 64] + [z, y, x]
                    save_indices = filter_coordinates_within_bb(save_indices, mask_bb)
                    center_of_mass = np.round(np.mean(save_indices, axis=0)).astype(int)

                    x = int(center_of_mass[2]); y = int(center_of_mass[1]); z = int(center_of_mass[0])

                    if check_overlap(mask, save_indices, mask_bb) > 70:
                        del_id.append(current_ind)
                        print(f"Tracking of cell {current_ind} is stopped. Overlap!")
                        continue
                    else:
                        queried_values = mask[tuple(save_indices.T)]
                        zero_mask = (queried_values == 0)
                        save_indices = save_indices[zero_mask]

                    if dataset == "TRIC":
                        if x <= 50 or y <= 50:
                            if not mitosis:
                                del_id.append(current_ind)
                                print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                                continue
                        if y >= image_y - 50 or x >= image_x - 50:
                            if not mitosis:
                                del_id.append(current_ind)
                                print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                                continue
                    else:
                        if x <= 5 or y <= 5 or z <= 2:
                            if not mitosis:
                                del_id.append(current_ind)
                                print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                                continue
                        if y >= image_y - 5 or x >= image_x - 5 or z >= image_z - 2:
                            if not mitosis:
                                del_id.append(current_ind)
                                print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                                continue

                    bb_points = get_prompt_bb_3d(save_indices, center_of_mass)
                    bb_points = np.array(bb_points) + np.array([y, x, y, x])

                    temp_dict[current_ind]['centroid'] = [z, y, x]
                    temp_dict[current_ind]['centroid_old'] = [z_old, y_old, x_old]
                    temp_dict[current_ind]['bb_points'] = bb_points

                    output_dict[current_ind]['end'] = tf
                    mask[tuple(save_indices.T)] = current_ind
                    continue

                # Compare features
                similarity = compare_features_generic(
                    sam2, [z, y, x], neighbor_positions, bb_points, images[tf-1:tf+1], tf,
                    current_ind, window_size, image_y, image_x, res_path, visualize
                )

                # Select top candidates with dataset-specific thresholds
                detected_centroids, similarity = process_lists(neighbor_positions, similarity, top_thresh, diff_thresh)

                if current_ind in new_born_ids:
                    detected_centroids = detected_centroids[:1]

                if len(detected_centroids) > 1:
                    z0, y0, x0 = map(int, detected_centroids[0])
                    z1, y1, x1 = map(int, detected_centroids[1])
                    value0 = mask[z0, y0, x0]
                    value1 = mask[z1, y1, x1]
                    if value0 != 0 and value1 == 0:
                        detected_centroids = [detected_centroids[1]]
                    elif value1 != 0 and value0 == 0:
                        detected_centroids = [detected_centroids[0]]
                    elif value0 != 0 and value1 != 0:
                        del_id.append(current_ind)
                        print(f"Tracking of cell {current_ind} is stopped. Both overlap!")
                        continue
                    else:
                        print(f"Mitosis detected for cell {current_ind}!")
                        print(current_ind)
                        print(similarity)
                        del_id.append(current_ind)
                        mitosis = True
                        parent = current_ind

                for detected_centroid in detected_centroids:
                    if mitosis:
                        current_ind = new_ind
                        new_ind += 1

                    z, y, x = map(int, detected_centroid)

                    new_array = segmenter.segmentation([z, y, x])
                    indices = np.argwhere(new_array == 1)

                    if indices.shape[0] <= min_voxels_general and not mitosis:
                        del_id.append(current_ind)
                        print(f"Tracking of cell {current_ind} is stopped. Mask is too small!")
                        continue

                    save_indices = indices - [64, 64, 64] + [z, y, x]
                    save_indices = filter_coordinates_within_bb(save_indices, mask_bb)
                    center_of_mass = np.round(np.mean(save_indices, axis=0)).astype(int)

                    x = int(center_of_mass[2]); y = int(center_of_mass[1]); z = int(center_of_mass[0])

                    if dataset == "TRIC":
                        if x <= 50 or y <= 50:
                            if not mitosis:
                                del_id.append(current_ind)
                                print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                                continue
                        if y >= image_y - 50 or x >= image_x - 50:
                            if not mitosis:
                                del_id.append(current_ind)
                                print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                                continue
                    else:
                        if x <= 5 or y <= 5 or z <= 2:
                            if not mitosis:
                                del_id.append(current_ind)
                                print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                                continue
                        if y >= image_y - 5 or x >= image_x - 5 or z >= image_z - 2:
                            if not mitosis:
                                del_id.append(current_ind)
                                print(f"Tracking of cell {current_ind} is stopped. Cell moves away!")
                                continue

                    if check_overlap(mask, save_indices, mask_bb) > 70 and not mitosis:
                        del_id.append(current_ind)
                        print(f"Tracking of cell {current_ind} is stopped. Overlap!")
                        continue
                    else:
                        queried_values = mask[tuple(save_indices.T)]
                        zero_mask = (queried_values == 0)
                        save_indices = save_indices[zero_mask]

                    bb_points = get_prompt_bb_3d(save_indices, center_of_mass)
                    bb_points = np.array(bb_points) + np.array([y, x, y, x])

                    if mitosis:
                        print(f"New cell {current_ind} saved")
                        temp_dict[current_ind] = {}
                        output_dict[current_ind] = {}
                        output_dict[current_ind]['start'] = tf
                        output_dict[current_ind]['parent'] = parent
                        new_born_ids.append(current_ind)

                    temp_dict[current_ind]['centroid'] = [z, y, x]
                    temp_dict[current_ind]['centroid_old'] = [z_old, y_old, x_old]
                    temp_dict[current_ind]['bb_points'] = bb_points

                    output_dict[current_ind]['end'] = tf
                    mask[tuple(save_indices.T)] = current_ind

            for id_ in del_id:
                del temp_dict[id_]

        added_pos = []
        for value in temp_dict.keys():
            if int(value) not in mask:
                add_x = temp_dict[value]['centroid'][2]
                add_y = temp_dict[value]['centroid'][1]
                add_z = temp_dict[value]['centroid'][0]
                print(f"Add dot of cell {int(value)}")
                while [add_z, add_y, add_x] in added_pos:
                    add_z -= 1
                    add_y -= 1
                    add_x -= 1
                mask[add_z, add_y, add_x] = value
                added_pos.append([add_z, add_y, add_x])

        tiff.imwrite(os.path.join(res_path, f"mask{tf:03d}.tif"), mask.astype(np.uint16))
        output_temp_json_file = os.path.join(res_path, 'temp_res_track.json')
        with open(output_temp_json_file, 'w') as json_file:
            json.dump(output_dict, json_file, indent=4)

    output_file = os.path.join(res_path, 'res_track.txt')
    with open(output_file, 'w') as f:
        for mask_id, mask_info in output_dict.items():
            f.write(f"{mask_id} {mask_info['start']} {mask_info['end']} {mask_info['parent']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM2Cetra with configurable parameters.")

    parser.add_argument("--dataset", type=str, required=True, choices=["TRIC", "TRIF"])
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--detection_path", type=str, required=True)
    parser.add_argument("--res_path", type=str, required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--init_json_path", type=int, required=True)
    parser.add_argument("--neighbor_thresh", type=int, required=True)
    parser.add_argument("--z_divisor", type=float, required=True)
    parser.add_argument("--top_thresh", type=float, required=True)
    parser.add_argument("--diff_thresh", type=float, required=True)
    parser.add_argument("--min_voxels_no_neighbor", type=int, required=True)
    parser.add_argument("--edge_guard_xy", type=int, required=True)
    parser.add_argument("--edge_guard_z_low", type=int, required=True)
    parser.add_argument("--edge_guard_z_high", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--min_voxels_general", type=int, default=1000)
    parser.add_argument("--neg_num", type=int, default=50)
    parser.add_argument("--visualize", default=False)
    parser.add_argument("--seed", type=int, default=2023)

    args = parser.parse_args()
    np.random.seed(args.seed)
    main(args)