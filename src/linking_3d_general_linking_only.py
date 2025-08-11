import os
import argparse
import tifffile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sam2.apply_sam2 import build_sam2
from utils.visualize import show_mask, show_points, show_box
from utils.tools import track_continuity_of_elements, find_closest_nonzero_and_keep_region_3d, get_prompt_points_3d, link_existing_cells, extract_tiff_3d, get_centriod, find_closest_nonzero_value_and_distance_3d

# Prediction shift in two adjacent frames
def get_shift(predictor, pos, pts_list, bb_points, img_list, reversed_ind, mask_ind, tmp_mask, size, image_y, image_x, neg_num, image_path):
    # Size of the 2D slice
    overlay_size = size
    half_overlay_size = overlay_size // 2

    x = pos[2]
    y = pos[1]
    z = pos[0]

    # Extract the 2D slice patch of the current and previous image
    start_x = x - half_overlay_size
    end_x = start_x + overlay_size
    start_y = y - half_overlay_size
    end_y = start_y + overlay_size 

    patches = [extract_tiff_3d(p, z, start_y, end_y, start_x, end_x, overlay_size, image_y, image_x) for p in reversed(img_list)]

    # Save the images as Img
    img_list = []
    for idx, img in enumerate(patches): 
        img_list.append(Image.fromarray(img.astype('uint8')))

    # Get the local position of the prompt points
    pts_list = pts_list - [start_y, start_x]
    # y, x to x, y, which is required by SAM2
    pts_list[:, [0, 1]] = pts_list[:, [1, 0]]
    bb_points = bb_points - [start_y, start_x, start_y, start_x]    
    #bb_points = bb_points[[1, 0, 3, 2]]

    ann_frame_idx = 0  # the frame index we interact with, here the first frame (current frame)
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    # concatenate fixed points to chosen points
    points = np.array([[half_overlay_size, half_overlay_size], [half_overlay_size, 6], [half_overlay_size, overlay_size-6], [6, half_overlay_size], [overlay_size-6, half_overlay_size]], dtype=np.float32)
    points = np.concatenate((points, pts_list.astype(np.float32)), axis=0)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 0, 0, 0, 0], np.int32)
    ones_list = [1] * 10
    zero_list = [0] * neg_num
    # Concatenate labels
    ones_array = np.array(ones_list+zero_list, np.int32)
    labels = np.concatenate((labels, ones_array))

    # Initialize SAM2
    inference_state = predictor.init_state_2(img_list=img_list)
    predictor.reset_state(inference_state)

    # If the last 2D mask is available, we use it to regulate the current mask
    if tmp_mask is not None:
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=tmp_mask,
        )
    
    # Add prompt bounding box
    box = np.array(bb_points, dtype=np.float32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )
    
    # Add prompt points 
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    '''
    # Visualize the prompted frame
    if visualize:
        plt.figure(figsize=(12, 8))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.fromarray(patches[ann_frame_idx]))
        #plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        #show_points(points, labels, plt.gca())
        show_box(box, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        mask_folder_name = out_path + f"/vis/masks{mask_ind:03d}"
        if not os.path.exists(mask_folder_name):
            os.makedirs(mask_folder_name)
        plt.savefig(mask_folder_name + f'/mask_{reversed_ind:03d}_p.png', dpi=300, bbox_inches='tight')
        plt.close()
    '''
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    # Get the shape of the mask, used for regularization of the next mask
    tmp_mask = video_segments[0][1]
    # Get the tracked mask in the previous time frame
    out_mask = video_segments[1][1]
    '''
    if visualize:
        plt.figure(figsize=(12, 8))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.fromarray(patches[-1]))
        #plt.imshow(Image.open(os.path.join(video_dir, frame_names[-1])))
        #show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        mask_folder_name = out_path + f"/vis/masks{mask_ind:03d}"
        if not os.path.exists(mask_folder_name):
            os.makedirs(mask_folder_name)
        plt.savefig(mask_folder_name + f'/mask_{reversed_ind:03d}_p2.png', dpi=300, bbox_inches='tight')
        plt.close()
    '''
    # Keep the connected region in the output mask
    out_mask = find_closest_nonzero_and_keep_region_3d(out_mask, (int(size/2), int(size/2)))    
    true_indices = np.argwhere(out_mask)
    # Return center of the tracked mask and the mask for regularization
    return np.mean(true_indices, axis=0), tmp_mask


def main(args): 
    # Initialize paths
    image_path = args.image_path
    mask_path = args.mask_path
    out_path = args.out_path

    size = args.size
    dis_threshold = args.dis_threshold
    neg_num = args.neg_num
    visualize = args.visualize
    neighbor_dist = args.neighbor_dist

    new_ind = 0

    # Extract paths of raw images and masks 
    images = []
    masks = []

    for path in sorted(os.listdir(image_path)):
        full_path = os.path.join(image_path, path)
        images.append(full_path)
    for path in sorted(os.listdir(mask_path)):
        full_path = os.path.join(mask_path, path)
        masks.append(full_path)

    # Get time frame number
    start_tf = 0
    end_tf = len(masks)

    images = images[start_tf:end_tf]
    masks = masks[start_tf:end_tf]

    # Get initial mask and image size
    init_mask = tifffile.imread(masks[-1])
    image_z = init_mask.shape[0]
    image_y = init_mask.shape[1]
    image_x = init_mask.shape[2]

    # Build SAM2
    sam2 = build_sam2()

    temp_dict = {} # dictionary for all tracked masks
    last_mask = None # last precessed mask
    mito_dict = {} # dictionary for mitosis information

    for i in range(start_tf, end_tf):
        # Backwards iteration
        reversed_index = len(images) - 1 - i
        print(f"Processing time frame {reversed_index:03d}")

        if i > 0:
            last_mask = mask

        # Initialize output mask	
        mask = np.zeros((image_z, image_y, image_x), dtype=np.uint16)
        
        # First time frame
        if i == 0:
            # Get initial positions from the last mask array
            positions, values = get_centriod(init_mask)

            # Iterate through all centroids of masks
            for current_ind, position in enumerate(positions):
                # Assign an index for the current cell
                current_ind += 1
                if current_ind % 50 == 0:
                    print(f"Tracking cell: {current_ind:03d}")
                # Get the cell position
                x = int(position[2])
                y = int(position[1])
                z = int(position[0])
                tmp_mask = None

                # Get mask pixels
                save_indices = np.argwhere(init_mask == values[current_ind-1])
                # Add mask to output file
                mask[tuple(save_indices.T)] = current_ind

                # Get positive / negative prompt points based on the mask
                slice_pts, bb_points = get_prompt_points_3d(save_indices, [z, y, x], size, neg_num)
                if slice_pts is not None:
                    # Transform the coordinates to global coordinates
                    slice_pts = slice_pts + [y, x]
                    bb_points = np.array(bb_points) + np.array([y, x, y, x])
                # Save the information of current cell and prepare for next iteration
                temp_dict[current_ind] = {}
                temp_dict[current_ind]['centroid'] = [z, y, x]
                temp_dict[current_ind]['slice_pts'] = slice_pts
                temp_dict[current_ind]['tmp_mask'] = tmp_mask
                temp_dict[current_ind]['centroid_old'] = [z, y, x] 
                temp_dict[current_ind]['bb_points'] = bb_points   
                if current_ind > new_ind:
                    new_ind = current_ind
        else:
            # Load new mask file and check the masks
            mask_arr = tifffile.imread(masks[reversed_index])
            existing_values = np.unique(mask_arr)
            existing_values = existing_values[1:]
            existing_values_dict = {value: [] for value in existing_values} # Get existing values of the mask file

            # Save indices of masks to be deleted 
            del_id = []
            # Dict to save new parent cells
            new_dict = {}
            # Iterate through all cells that are saved
            for current_ind in temp_dict.keys():
                if current_ind % 50 == 0:
                    print(f"Tracking cell: {current_ind:03d}")
                # Get click position
                z = temp_dict[current_ind]['centroid'][0]
                y = temp_dict[current_ind]['centroid'][1]
                x = temp_dict[current_ind]['centroid'][2]
                # Get prompt points (as global position)
                slice_pts = temp_dict[current_ind]['slice_pts'] 
                bb_points = temp_dict[current_ind]['bb_points'] 
                # Get previous mask for regularization
                tmp_mask = temp_dict[current_ind]['tmp_mask'] 
                
                # Save the previous position before it's updated
                x_old = x
                y_old = y
                z_old = z
                # Get shift vector based on SAM2
                if slice_pts is not None:
                    shift_vec, tmp_mask = get_shift(sam2, [z, y, x], slice_pts, bb_points, images[reversed_index:reversed_index+2], reversed_index, current_ind, tmp_mask, size, image_y, image_x, neg_num, image_path)
                    
                    # When the shift vector is invalid, we set the shift to 0
                    if np.isnan(shift_vec).any():
                        print(f"Cell {current_ind}: not tracked, using old centroid")
                        shift_vec = [size/2, size/2, size/2]       

                    # Shift the centroid, update the click position
                    x = x + int(shift_vec[2] - size/2)
                    y = y + int(shift_vec[1] - size/2)
                    z = z
                # Get the value at the predicted centroid
                closest_value, distance = find_closest_nonzero_value_and_distance_3d(mask_arr, (z, y, x))
                if distance <= neighbor_dist:
                    value = closest_value # Mask is linked 
                                    # If directly linked, just use the linked mask
                    save_indices = np.argwhere(mask_arr == value)
                    center_of_mass = np.round(np.mean(save_indices, axis=0)).astype(int)
                else:
                    value = 0 # Mask is not linked
    

                # If there's a linked mask, check if there is mitosis, same as 2D
                if value != 0:
                    if len(existing_values_dict[value]) == 0:
                        existing_values_dict[value].append(current_ind)
                    elif len(existing_values_dict[value]) == 1:
                        existing_values_dict[value].append(current_ind)
                        del_id.append(existing_values_dict[value][0])
                        del_id.append(current_ind)
                        mask[mask==existing_values_dict[value][0]] = 0
                        new_ind += 1
                        mito_dict[existing_values_dict[value][0]] = {}
                        mito_dict[existing_values_dict[value][0]]["parent"] = new_ind
                        mito_dict[existing_values_dict[value][0]]["first_tf"] = reversed_index+1
                        mito_dict[current_ind] = {}
                        mito_dict[current_ind]["parent"] = new_ind
                        mito_dict[current_ind]["first_tf"] = reversed_index+1
                        print(f"{current_ind} merged with {existing_values_dict[value][0]} to {new_ind}!")
                        current_ind = new_ind
                        slice_pts, bb_points = get_prompt_points_3d(save_indices, center_of_mass, size, neg_num)
                        if slice_pts is not None:
                            # Transform the coordinates to global coordinates
                            slice_pts = slice_pts + [y, x]
                            bb_points = np.array(bb_points) + np.array([y, x, y, x])
                        new_dict[new_ind] = {}
                        new_dict[new_ind]['centroid'] = center_of_mass
                        new_dict[new_ind]['centroid_old'] = [z_old, y_old, x_old]                
                        new_dict[new_ind]['slice_pts'] = slice_pts
                        new_dict[new_ind]['tmp_mask'] = tmp_mask
                        new_dict[new_ind]['bb_points'] = bb_points   
                    else:
                        del_id.append(current_ind)
                else:
                    print(f"Tracking of cell {current_ind} is not linked!") # If no mask linked, just keep the mask generated by SAM3D       
                    del_id.append(current_ind)
                    continue
                    
                mask[tuple(save_indices.T)] = current_ind

                # Update the centroid of the mask
                x = int(center_of_mass[2])
                y = int(center_of_mass[1])
                z = round(center_of_mass[0])
                
                # Save information of non-parent cells
                if current_ind not in new_dict:
                    # Get prompt points (local coordinates)
                    slice_pts, bb_points = get_prompt_points_3d(save_indices, center_of_mass, size, neg_num)
                    if slice_pts is not None:
                        # Transform the coordinates to global coordinates
                        slice_pts = slice_pts + [y, x]
                        bb_points = np.array(bb_points) + np.array([y, x, y, x])
                    # Update the information of the current cell in dictionary
                    temp_dict[current_ind]['centroid'] = [z, y, x]
                    temp_dict[current_ind]['centroid_old'] = [z_old, y_old, x_old]                
                    temp_dict[current_ind]['slice_pts'] = slice_pts
                    temp_dict[current_ind]['tmp_mask'] = tmp_mask
                    temp_dict[current_ind]['bb_points'] = bb_points 

            # Delete cells that are not to be tracked anymore from the dictionary
            for id in del_id:
                del temp_dict[id]
            # Merge new parent cells
            if new_dict:
                temp_dict.update(new_dict)
            
            # Check existing masks that are not linked  
            for key, value in existing_values_dict.items():
                if value == []: # Unlinked mask
                    save_indices = np.argwhere(mask_arr == key)
                    center_of_mass = np.round(np.mean(save_indices, axis=0)).astype(int)
                    # Ignore small masks
                    if save_indices.shape[0] <= dis_threshold:
                        del_id.append(current_ind)
                        print(f"Tracking of a new cell is stopped. Mask is too small!")
                        continue
                    x = int(center_of_mass[2])
                    y = int(center_of_mass[1])
                    z = int(center_of_mass[0])
                    # Get the value at the same position in the previous mask, try to re-link
                    last_value = last_mask[z, y, x]

                    # If there was a mitosis, which might be wrong, we delete the mitosis and relink
                    if last_value in mito_dict.keys():
                        current_ind = last_value
                        print(f"new cell is linked to merged cell {last_value}!")  
                        # Find the wrong parent cell
                        wroung_cell = mito_dict[last_value]["parent"]
                        # Find the daughter cells and relink to the original value
                        del mito_dict[last_value]
                        for search_id, search_dict in mito_dict.items():
                            if search_dict["parent"] == wroung_cell:
                                break
                        # Delete the wrong parent cell
                        del mito_dict[search_id]
                        temp_dict[search_id] = temp_dict[wroung_cell]
                        del temp_dict[wroung_cell]
                        # Relink
                        mask[mask==wroung_cell] = search_id
                    elif last_value not in temp_dict.keys() and last_value != 0: # Link to a deleted cell
                        print(f"new cell is linked to disappeared cell {last_value}")
                        current_ind = last_value
                    else: # Generate a new cell and assign a new label
                        new_ind += 1
                        current_ind = new_ind
                        print(f"new cell {new_ind} detected!")  

                    mask[tuple(save_indices.T)] = current_ind
                    # Get prompt points (local coordinates)
                    slice_pts, bb_points = get_prompt_points_3d(save_indices, center_of_mass, size, neg_num)
                    if slice_pts is not None:
                        # Transform the coordinates to global coordinates
                        slice_pts = slice_pts + [y, x]
                        bb_points = np.array(bb_points) + np.array([y, x, y, x])
                    # Update the information of the current cell in dictionary
                    temp_dict[current_ind] = {}
                    temp_dict[current_ind]['centroid'] = [z, y, x]
                    temp_dict[current_ind]['centroid_old'] = [z, y, x]             
                    temp_dict[current_ind]['slice_pts'] = slice_pts
                    temp_dict[current_ind]['tmp_mask'] = None
                    temp_dict[current_ind]['bb_points'] = bb_points   
            
        # Save the result of the current time frame to file
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Add intermediate points (forced relinking)
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
        
        # Save output
        tifffile.imwrite(os.path.join(out_path,f"mask{reversed_index:03d}.tif"), mask)
    
    # Generate tracking file
    index_tracker = track_continuity_of_elements(start_tf, end_tf, out_path)

    output_file = os.path.join(out_path, 'res_track.txt')
    with open(output_file, 'w') as f:
        for number, (first, last) in index_tracker.items():
            if int(number) in mito_dict.keys():
                parent = mito_dict[int(number)]["parent"]
                f.write(f"{number} {first} {last} {parent}\n")
            else: 
                f.write(f"{number} {first} {last} 0\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM2Cetra with configurable parameters.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the raw images.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the existing masks.")
    parser.add_argument("--out_path", type=str, required=True, help="Result path.")
    parser.add_argument("--size", type=int, required=True, help="Local window size for SAM2.")
    parser.add_argument("--dis_threshold", type=int, required=True, help="Threshold for disappearing object.")
    parser.add_argument("--neighbor_dist", type=int, required=True, help="Distance threshold for linking masks.")
    parser.add_argument("--neg_num", type=int, default=50, help="Number of negative prompt points (fixed).")
    parser.add_argument("--visualize", type=bool, default=False, help="Visualize results (always deactivated for submission).")  
    parser.add_argument("--seed", type=int, default=2023, help="Seed")  

    args = parser.parse_args()

    np.random.seed(args.seed)
    
    main(args)
