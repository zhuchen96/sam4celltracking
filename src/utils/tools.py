import cv2
import tifffile
import numpy as np
from scipy import ndimage
from skimage import exposure
from collections import Counter
from scipy.ndimage import distance_transform_edt
import zarr

# Extract patch from tiff file
def extract_tiff(tiff_path, start_y, end_y, start_x, end_x, overlay_size, image_y, image_x):
    file = tifffile.imread(tiff_path)
    # Calculate the actual data to be extracted from the file
    actual_start_x = max(start_x, 0)
    actual_end_x = min(end_x, image_x)
    actual_start_y = max(start_y, 0)
    actual_end_y = min(end_y, image_y)
    
    # Extract the patch
    patch = file[actual_start_y:actual_end_y, actual_start_x:actual_end_x] 
    if "Fluo-N2DL-HeLa" in tiff_path:
        patch = exposure.rescale_intensity(patch, out_range=(0, 1))
        patch = exposure.adjust_gamma(patch, gamma=0.5)
    elif "PhC-C2DL-PSC" in tiff_path:
        patch = exposure.rescale_intensity(patch, out_range=(0, 1))
        patch = exposure.adjust_gamma(patch, gamma=1)
    elif "Fluo-N2DH-GOWT1" in tiff_path:
        patch = exposure.rescale_intensity(patch, out_range=(0, 1))
        patch = ndimage.median_filter(patch, size=5)
        patch = exposure.adjust_gamma(patch, gamma=0.1)
    elif "Fluo-N2DH-SIM+" in tiff_path:
        patch = exposure.rescale_intensity(patch, out_range=(0, 1))
        patch = ndimage.median_filter(patch, size=5)
        patch = exposure.adjust_gamma(patch, gamma=0.1)

    patch = exposure.rescale_intensity(patch, out_range=(0, 255))

    # Initialize a zero-padded patch of the desired overlay size
    padded_patch = np.zeros((overlay_size, overlay_size), dtype=patch.dtype)
    
    # Calculate where in the padded patch the real data will go
    pad_start_x = actual_start_x - start_x
    pad_start_y = actual_start_y - start_y
    
    # Insert the extracted patch into the padded patch
    padded_patch[pad_start_y:pad_start_y + patch.shape[0], pad_start_x:pad_start_x + patch.shape[1]] = patch
    
    padded_patch = padded_patch.astype(np.uint8)
    return padded_patch

# Get 2D prompt bounding box
def get_prompt_box(save_indices, center):
    try:
        save_indices = save_indices - center
        x_min = np.min(save_indices[:, 0])
        y_min = np.min(save_indices[:, 1])
        x_max = np.max(save_indices[:, 0])
        y_max = np.max(save_indices[:, 1])

        return [x_min, y_min, x_max, y_max]
    except:
        return None

# Get 3D prompt bounding box
def get_prompt_bb_3d(save_indices, center):
    try:
        # Local coordinates
        indices = save_indices - center
        first_column = indices[:, 0]
        #Find largest z-slice of each mask
        unique_values, counts = np.unique(first_column, return_counts=True)
        slice = unique_values[np.argmax(counts)].item()
        slice_rows = indices[indices[:, 0] == slice]

        # Get bounding box
        y_min = np.min(slice_rows[:, 1])
        x_min = np.min(slice_rows[:, 2])
        y_max = np.max(slice_rows[:, 1])
        x_max = np.max(slice_rows[:, 2])
        
        return [x_min, y_min, x_max, y_max]
    except:
        return None

# Get 2D prompt points
def get_prompt_points(neg_num, size, save_indices, center):
    try:
        indices = save_indices - center
        # Selete positive prompt points
        slice_rows = indices
        if slice_rows.shape[0] >= 10:
            slice_pts = slice_rows[np.random.choice(slice_rows.shape[0], 10, replace=False)]
        else:
            slice_pts = indices[np.random.choice(indices.shape[0], 10, replace=True)]

        existing_rows = set(map(tuple, slice_rows))

        # Initialize an empty list 
        neg_pts = np.empty((0, 2), dtype=int)
        # Generate new negative prompt points
        while len(neg_pts) < neg_num:
            new_rows = np.random.randint(-size/2+3, size/2-3, size=(50, 2))
            # Keep only unique points that are not already in existing list of points
            new_rows = np.array([tuple(row) for row in new_rows if tuple(row) not in existing_rows])
            
            # Add valid unique points to neg_pts
            neg_pts = np.vstack((neg_pts, new_rows[:neg_num - len(neg_pts)]))
            
            # Update existing rows set with the new points
            existing_rows.update(map(tuple, new_rows))
        
        # Concatenate the selected positive points and the generated negative points
        slice_pts = np.vstack((slice_pts, neg_pts))
        
        return slice_pts
    except:
        return None

# Get centroids of masks for a given file
def get_centriod(mask):
    centroids = []
    mask_values = []
    
    for mask_value in np.unique(mask):
        if mask_value == 0:  # Skip background
            continue
        # Get coordinates of all pixels in the current mask
        coords = np.argwhere(mask == mask_value)
        # Calculate the centroid of the current mask
        centroid = np.mean(coords, axis=0)
        centroids.append(centroid)
        mask_values.append(mask_value)
    
    # Convert centroids to array
    centroids = np.array(centroids)
    return centroids, mask_values

# Keep one connected region of the generated mask
def find_closest_nonzero_and_keep_region_3d(array, centroid):
    assert array.ndim == 3 and array.shape[0] == 1, "Input array must have shape (1, H, W)."
    assert len(centroid) == 2, "Centroid must be a tuple (y, x)."
    
    # Extract the 2D array from the 3D array
    array_2d = array[0]
    
    # Convert array to uint8
    array_2d = array_2d.astype(np.uint8)

    # Compute the distance transform to find the closest non-zero pixel
    distances, indices = distance_transform_edt(array_2d == 0, return_indices=True)
    closest_y, closest_x = indices[0, centroid[0], centroid[1]], indices[1, centroid[0], centroid[1]]

    # Check if the closest pixel is non-zero
    if array_2d[closest_y, closest_x] == 0:
        return np.zeros_like(array, dtype=np.uint8)  # No non-zero pixel found

    # Use connected components to isolate the region containing the closest pixel
    num_labels, labels = cv2.connectedComponents(array_2d, connectivity=8)

    # Get the label of the closest pixel
    closest_pixel_label = labels[closest_y, closest_x]

    # Create a mask for the connected region of the closest pixel
    region_mask_2d = (labels == closest_pixel_label).astype(np.uint8)

    # Expand dimensions to match the original array shape (1, H, W)
    region_mask_3d = region_mask_2d[np.newaxis, ...]

    return region_mask_3d

# Find the closest non-zero value in a 2D array to a given coordinate and its distance.
def find_closest_nonzero_value_and_distance(array, coordinate):
    assert array.ndim == 2, "Input array must be 2D."
    assert len(coordinate) == 2, "Coordinate must be a tuple (y, x)."
    
    # Compute the distance transform and the indices of closest non-zero pixels
    distances, indices = distance_transform_edt(array == 0, return_indices=True)
    
    # Get the closest pixel's coordinates
    closest_y, closest_x = indices[0, coordinate[0], coordinate[1]], indices[1, coordinate[0], coordinate[1]]
    
    # Get the value of the closest non-zero pixel
    closest_value = array[closest_y, closest_x]
    
    # Get the distance to the closest non-zero pixel
    distance = distances[coordinate[0], coordinate[1]]
    
    return closest_value, distance

# Check tracked cell
def track_continuity_of_elements(start_tf, end_tf, check_dir):
    index_tracker = {}  # Dictionary to store first and last occurrences of each integer
    seen_in_previous = set()  # Track elements seen in the previous time frame

    # Loop through each time frame mask file
    for index in range(start_tf, end_tf):
        # Read mask file
        mask = tifffile.imread(f'{check_dir}/mask{index:03d}.tif')
        unique_values = set(np.unique(mask))  # Convert to set for faster operations
        
        # Check for continuity
        for number in unique_values:
            if number == 0:
                continue  # Skip background or undefined regions if zero is background
            
            # Update the index_tracker with the first and last appearance of each number
            if number in index_tracker:
                # Update last occurrence index
                index_tracker[number] = (index_tracker[number][0], index)
            else:
                # First appearance, initialize with current index for both first and last
                index_tracker[number] = (index, index)

            # Check if the number reappears after a gap (skip if this is its first time)
            if number in index_tracker and number not in seen_in_previous and index_tracker[number][0] != index:
                print(f"Error: Element {number} reappeared at time frame {index} after disappearing.")
        
        # Update seen_in_previous to the current unique values for the next iteration check
        seen_in_previous = unique_values

    return index_tracker

def filter_mask(arr1, arr2):
    # Extract the first two columns from both arrays
    arr1_first_two = arr1[:, 1:]
    arr2_first_two = arr2[:, 1:]

    # Use broadcasting to compare all rows in arr1_first_two with arr2_first_two
    mask = np.any(np.all(arr1_first_two[:, None] == arr2_first_two, axis=-1), axis=-1)

    # Apply the mask to filter out rows in arr1
    filtered_arr1 = arr1[mask]

    return filtered_arr1

# Get 3D prompt points and bounding box
def get_prompt_points_3d(save_indices, center, size, neg_num):
    try:
        # Local coordinates
        indices = save_indices - center
        first_column = indices[:, 0]
        #Find largest z-slice of each mask
        unique_values, counts = np.unique(first_column, return_counts=True)
        slice = unique_values[np.argmax(counts)].item()
        slice_rows = indices[indices[:, 0] == slice]

        # Get bounding box
        y_min = np.min(slice_rows[:, 1])
        x_min = np.min(slice_rows[:, 2])
        y_max = np.max(slice_rows[:, 1])
        x_max = np.max(slice_rows[:, 2])

        # Get positive prompt points
        if slice_rows.shape[0] >= 10:
            slice_pts = slice_rows[np.random.choice(slice_rows.shape[0], 10, replace=False)]
        else:
            slice_pts = indices[np.random.choice(indices.shape[0], 10, replace=False)]
        slice_pts = slice_pts[:,1:]
        existing_rows = set(map(tuple, slice_rows[:,1:]))

        # Initialize an empty list 
        neg_pts = np.empty((0, 2), dtype=int)
        # Get negative prompt points
        while len(neg_pts) < neg_num:
            # Generate enough random points
            new_rows = np.random.randint(-size/2+3, size/2-3, size=(50, 2))
            # Keep only unique points 
            new_rows = np.array([tuple(row) for row in new_rows if tuple(row) not in existing_rows])
            neg_pts = np.vstack((neg_pts, new_rows[:neg_num - len(neg_pts)]))
            existing_rows.update(map(tuple, new_rows))
        
        # Concatenate the positive slice points and the generated negative points
        slice_pts = np.vstack((slice_pts, neg_pts))
        
        return slice_pts, [x_min, y_min, x_max, y_max]
    except:
        return None, None

# Link new mask to existing mask
def link_existing_cells(last_mas_arr, save_indices, image_z, image_y, image_x):
    save_indices = np.array(save_indices)  # Ensure save_indices is a numpy array
    ranges = [(0, image_z-1), (0, image_y-1), (0, image_x-1)]  # Define the valid ranges for each column

    # Remove pixels that are out of boundary
    mask = np.all(
        (save_indices >= np.array([r[0] for r in ranges])) &
        (save_indices <= np.array([r[1] for r in ranges])),
        axis=1
    )
    save_indices = save_indices[mask]

    current_values = last_mas_arr[tuple(save_indices.T)]  # Get existing values at the coordinates
    
    value_counts = Counter(current_values)

    # Find the most common value and its count
    most_common_value, most_common_count = value_counts.most_common(1)[0]

    # Get the most common value: this should be the linked mask
    if most_common_value != 0:
        out_ind = np.argwhere(last_mas_arr==most_common_value)
        # Take the linked mask
        return out_ind, np.round(np.mean(out_ind, axis=0)).astype(int), most_common_value
    else:
        # Use the original mask if it's not linked
        return save_indices, np.round(np.mean(save_indices, axis=0)).astype(int), 0

# Extract 3D patch from a tiff file
def extract_tiff_3d(tiff_path, z, start_y, end_y, start_x, end_x, overlay_size, image_y, image_x):
    file = tifffile.imread(tiff_path)
    # Calculate the actual data to be extracted from the file
    actual_start_x = max(start_x, 0)
    actual_end_x = min(end_x, image_x)
    actual_start_y = max(start_y, 0)
    actual_end_y = min(end_y, image_y)
    
    # Extract the patch
    patch = file[z, actual_start_y:actual_end_y, actual_start_x:actual_end_x]

    if "Fluo-N3DH-SIM+" in tiff_path:
        patch = exposure.rescale_intensity(patch, out_range=(0, 1))
        patch = ndimage.median_filter(patch, size=5)
        patch = exposure.adjust_gamma(patch, gamma=0.1)

    patch = exposure.rescale_intensity(patch, out_range=(0, 255))
    # Initialize a zero-padded patch of the desired overlay size
    padded_patch = np.zeros((overlay_size, overlay_size), dtype=patch.dtype)
    
    # Calculate where in the padded patch the real data will go
    pad_start_x = actual_start_x - start_x
    pad_start_y = actual_start_y - start_y
    
    # Insert the extracted patch into the padded patch
    padded_patch[pad_start_y:pad_start_y + patch.shape[0], pad_start_x:pad_start_x + patch.shape[1]] = patch

    padded_patch = padded_patch.astype(np.uint8)
    return padded_patch

def find_closest_nonzero_value_and_distance_3d(array, coordinate):
    assert array.ndim == 3, "Input array must be 3D."
    assert len(coordinate) == 3, "Coordinate must be a tuple (z, y, x)."
    
    # Compute the distance transform and indices of the closest non-zero pixels
    distances, indices = distance_transform_edt(array == 0, return_indices=True)
    
    # Get the closest pixel's coordinates
    closest_z = indices[0, coordinate[0], coordinate[1], coordinate[2]]
    closest_y = indices[1, coordinate[0], coordinate[1], coordinate[2]]
    closest_x = indices[2, coordinate[0], coordinate[1], coordinate[2]]
    
    # Get the value of the closest non-zero pixel
    closest_value = array[closest_z, closest_y, closest_x]
    
    # Get the distance to the closest non-zero pixel
    distance = distances[coordinate[0], coordinate[1], coordinate[2]]
    
    return closest_value, distance

def extract_zarr_3d(zarr_path, z, start_y, end_y, start_x, end_x, overlay_size, image_y, image_x):
    image_arr = zarr.open(zarr_path, mode='r')
    

    # Calculate the actual data to be extracted from the file
    actual_start_x = max(start_x, 0)
    actual_end_x = min(end_x, image_x)
    actual_start_y = max(start_y, 0)
    actual_end_y = min(end_y, image_y)
    
    # Extract the patch
    patch = image_arr[z, actual_start_y:actual_end_y, actual_start_x:actual_end_x]

    patch = exposure.rescale_intensity(patch, out_range=(0, 255))
    # Initialize a zero-padded patch of the desired overlay size
    padded_patch = np.zeros((overlay_size, overlay_size), dtype=patch.dtype)
    
    # Calculate where in the padded patch the real data will go
    pad_start_x = actual_start_x - start_x
    pad_start_y = actual_start_y - start_y
    
    # Insert the extracted patch into the padded patch
    padded_patch[pad_start_y:pad_start_y + patch.shape[0], pad_start_x:pad_start_x + patch.shape[1]] = patch

    padded_patch = padded_patch.astype(np.uint8)
    return padded_patch

def extract_zarr_3d_mip(zarr_path, z, start_y, end_y, start_x, end_x, overlay_size, image_z, image_y, image_x):
    image_arr = zarr.open(zarr_path, mode='r')
    

    # Calculate the actual data to be extracted from the file
    actual_start_x = max(start_x, 0)
    actual_end_x = min(end_x, image_x)
    actual_start_y = max(start_y, 0)
    actual_end_y = min(end_y, image_y)
    actual_start_z = max(z-2, 0)
    actual_end_z = max(z+3, image_z)
    
    if actual_start_z == actual_end_z:
        patch = image_arr[z, actual_start_y:actual_end_y, actual_start_x:actual_end_x]
    else:
        # Extract the patch
        patch = image_arr[actual_start_z:actual_end_z, actual_start_y:actual_end_y, actual_start_x:actual_end_x]
        patch = np.max(patch, axis=0)
    patch = exposure.rescale_intensity(patch, out_range=(0, 255))
    # Initialize a zero-padded patch of the desired overlay size
    padded_patch = np.zeros((overlay_size, overlay_size), dtype=patch.dtype)
    
    # Calculate where in the padded patch the real data will go
    pad_start_x = actual_start_x - start_x
    pad_start_y = actual_start_y - start_y
    
    # Insert the extracted patch into the padded patch
    padded_patch[pad_start_y:pad_start_y + patch.shape[0], pad_start_x:pad_start_x + patch.shape[1]] = patch

    padded_patch = padded_patch.astype(np.uint8)
    return padded_patch
