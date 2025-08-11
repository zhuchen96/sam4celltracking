import os
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
import numpy as np
import tifffile
from sam_med3d.build_sam3D import sam_model_registry3D

class Segmenter:
    def __init__(self, ind, folder, img_z, img_y, img_x, model):
        self.seed = 2023
        #print("set seed as", SEED)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.init()
        
        # Load sam3d model
        checkpoint_path = model
        self.device = 'cuda'
        self.model_type = 'vit_b_ori'
        self.crop_size = 128 # Here we have a fixed size for the 3D patch

        self.sam_model_tune = sam_model_registry3D[self.model_type](checkpoint=None).to(self.device)
        if checkpoint_path is not None:
            model_dict = torch.load(checkpoint_path, map_location=self.device)
            state_dict = model_dict['model_state_dict']
            self.sam_model_tune.load_state_dict(state_dict)
        self.sam_model_tune.eval()
        self.images = []
        img_folder = folder
        for path in sorted(os.listdir(img_folder)):
            full_path = os.path.join(img_folder, path)
            self.images.append(full_path)

        self.ind = ind

        self.image_x = img_x
        self.image_y = img_y
        self.image_z = img_z

    def update_image(self, ind):
        SEED = self.seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.ind = ind
        
    def segmentation(self, coord_list):
        overlay_size = 128
        half_overlay_size = overlay_size // 2

        x = coord_list[2]
        y = coord_list[1]
        z = coord_list[0]

        start_x = x - half_overlay_size
        end_x = start_x + overlay_size
        start_y = y - half_overlay_size
        end_y = start_y + overlay_size 
        start_z = z - half_overlay_size
        end_z = start_z + overlay_size 
        start_x = max(start_x, 0)
        start_y = max(start_y, 0)
        start_z = max(start_z, 0)
        end_x = min(end_x, self.image_x)
        end_y = min(end_y, self.image_y)
        end_z = min(end_z, self.image_z)

        sitk_image_arr = extract_tiff(self.images[self.ind], start_z, end_z, start_y, end_y, start_x, end_x)
        sitk_image_arr = sitk_image_arr.astype(np.float32)
        self.sitk_image_arr = sitk_image_arr.reshape((1,) + sitk_image_arr.shape)
        subject = tio.Subject(
            image = tio.ScalarImage(tensor=self.sitk_image_arr),
        )
        crop_mask = torch.zeros_like(subject.image.data)

        prompt_position = [coord_list[0] - start_z, coord_list[1] - start_y, coord_list[2] - start_x]
        random_index = [0] + prompt_position
        # print(crop_mask.shape)
        crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1

        subject.add_image(tio.LabelMap(tensor=crop_mask),
                            image_name="crop_mask")
        subject = tio.CropOrPad(mask_name='crop_mask', 
                                target_shape=(128,128,128))(subject)

        image3D = subject.image.data.unsqueeze(0)
        sz = image3D.size()

        norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

        img3D = norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
        img3D = img3D.unsqueeze(dim=1)

        with torch.no_grad():
            image_embedding = self.sam_model_tune.image_encoder(img3D.to(self.device)) # (1, 384, 16, 16, 16)

        prev_masks = torch.zeros_like(image3D).to(self.device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(self.crop_size//4,self.crop_size//4,self.crop_size//4))
        
        sparse_embeddings, dense_embeddings = self.sam_model_tune.prompt_encoder(
                points=[torch.tensor([64, 64, 64]).unsqueeze(0).unsqueeze(0).to(self.device), torch.tensor([1]).unsqueeze(0).to(self.device)],
                boxes=None,
                masks=low_res_masks.to(self.device),
            )
        low_res_masks, _ = self.sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(self.device), # (B, 384, 64, 64, 64)
            image_pe=self.sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
            dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
            multimask_output=False,
            )
        prev_masks = F.interpolate(low_res_masks, size=image3D.shape[-3:], mode='trilinear', align_corners=False)

        medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
        # convert prob to mask
        medsam_seg_prob = medsam_seg_prob.detach().cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        return medsam_seg

def extract_tiff(tiff_path, start_z, end_z, start_y, end_y, start_x, end_x):
    file = tifffile.imread(tiff_path)
    patch = file[start_z:end_z, start_y:end_y, start_x:end_x]
    return patch
