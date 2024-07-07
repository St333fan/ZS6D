import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import argparse
import json
import os
import torch
from tqdm import tqdm
import numpy as np
from src.pose_extractor import PoseViTExtractor
from pose_utils.data_utils import ImageContainer_masks
import pose_utils.img_utils as img_utils
from PIL import Image
import cv2
import pose_utils.utils as utils
import pose_utils.vis_utils as vis_utils
import time
import pose_utils.eval_utils as eval_utils
import csv
import logging
import croco_match

# Setup logging
logging.basicConfig(level=logging.INFO, filename="pose_estimation.log",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Test pose estimation inference on test set')
    parser.add_argument('--config_file', default="./zs6d_configs/bop_eval_configs/cfg_ycbv_inference_bop_myset.json")

    args = parser.parse_args()

    with open(os.path.join(args.config_file), 'r') as f:
        config = json.load(f)
    
    # Loading ground truth files:

    with open(os.path.join(config['templates_gt_path']), 'r') as f:
        templates_gt = json.load(f)
    
    with open(os.path.join(config['gt_path']), 'r') as f:
        data_gt = json.load(f)
    
    with open(os.path.join(config['norm_factor_path']), 'r') as f:
        norm_factors = json.load(f)


    # Set up a results csv file:
    csv_file = os.path.join('./results', config['results_file'])

    # Column names for the CSV file
    headers = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']

    # Create a new CSV file and write the headers
    with open(csv_file, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)
        
    if config['debug_imgs']:
        debug_img_path = os.path.join("./debug_imgs",config['results_file'].split(".csv")[0])
        if not os.path.exists(debug_img_path):
            os.makedirs(debug_img_path)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    extractor = PoseViTExtractor(model_type='dino_vits8', stride=4, device=device)
    print("Loading PoseViTExtractor is done!")

    matches = []

    print("Processing input images:")
    for all_id, img_labels in tqdm(data_gt.items()):

        # enter the image which should be checked
        if all_id != '000059_1':
            continue

        scene_id = all_id.split("_")[0]
        img_id = all_id.split("_")[-1]
        
        # get data and crops for a single image
        img_path = os.path.join(config['dataset_path'], img_labels[0]['img_name'].split("./")[-1])
        img_name = img_path.split("/")[-1].split(".png")[0]

        img = Image.open(img_path)
        cam_K = np.array(img_labels[0]['cam_K']).reshape((3,3))

        img_data = ImageContainer_masks(img = img,
                                  img_name = img_name,
                                  scene_id = scene_id,
                                  cam_K = cam_K, 
                                  crops = [],
                                  descs = [],
                                  x_offsets = [],
                                  y_offsets = [],
                                  obj_names = [],
                                  obj_ids = [],
                                  model_infos = [],
                                  t_gts = [],
                                  R_gts = [],
                                  masks = [])

        for obj_index, img_label in enumerate(img_labels):
            bbox_gt = img_label[config['bbox_type']]

            if bbox_gt[2] == 0 or bbox_gt[3] == 0:
                continue

            if bbox_gt != [-1,-1,-1,-1]:
                img_data.t_gts.append(np.array(img_label['cam_t_m2c']) * config['scale_factor'])
                img_data.R_gts.append(np.array(img_label['cam_R_m2c']).reshape((3,3)))
                img_data.obj_ids.append(str(img_label['obj_id']))
                img_data.model_infos.append(img_label['model_info'])

                try:
                    mask = img_utils.rle_to_mask(img_label['mask_sam'])

                    mask = mask.astype(np.uint8)

                    mask_3_channel = np.stack([mask] * 3, axis=-1)

                    bbox = img_utils.get_bounding_box_from_mask(mask)

                    img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(img), bbox)

                    mask_crop,_,_ = img_utils.make_quadratic_crop(mask, bbox)

                    img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)

                    img_data.crops.append(Image.fromarray(img_crop))

                    img_prep, img_crop,_ = extractor.preprocess(Image.fromarray(img_crop), load_size=224)

                    mask_array = [
                        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
                    ]

                    assets_folder = '/home/stefan/PycharmProjects/ZS6D/templates/ycbv_desc/'+'obj_'+ str(img_label['obj_id'])

                    croco_match.process(segmented_image=img_crop,
                                        ckpt_path='/home/stefan/PycharmProjects/ZS6D/pretrained_models/CroCo.pth',
                                        # _V2_ViTLarge_BaseDecoder
                                        output_folder='/home/stefan/PycharmProjects/ZS6D/assets_match/decoded_images',
                                        assets_folder=assets_folder,
                                        mask_array=mask_array)

                    best_match = croco_match.find_match(segmented_image=img_crop,
                                    decoded_images_dir='/home/stefan/PycharmProjects/ZS6D/assets_match/decoded_images',
                                    mask_array=mask_array)

                    best_match = best_match.replace("decoded_", "")
                    best_temp = Image.open('/home/stefan/PycharmProjects/ZS6D/templates/ycbv_desc/'+'obj_'+
                                      str(img_label['obj_id']) +'/'+ best_match)

                    plt.imshow(best_temp)
                    plt.show()

                    print(best_match)
                    matches.append(all_id+'|'+ str(img_label['obj_id']) +'|'+best_match)

                    img_data.y_offsets.append(y_offset)
                    img_data.x_offsets.append(x_offset)
                    img_data.masks.append(mask_3_channel)

                except Exception as e:
                    matches.append(all_id + '|' + str(img_label['obj_id']) + '|' + '000000.png')
                    print(f"Warning: 'mask_sam' not found or bad defined in img_label. Skipping this iteration.")
                    logger.warning(f"Loading mask and extracting descriptor failed for img {img_path} and object_id {obj_index}: {e}")
                    img_data.crops.append(None)
                    img_data.descs.append(None)
                    img_data.y_offsets.append(None)
                    img_data.x_offsets.append(None)
                    img_data.masks.append(None)

        print(matches)
        break