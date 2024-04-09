from zs6d import ZS6D
import os
import json
from croco.models.croco import CroCoNet
import cv2
from PIL import Image
import pose_utils.img_utils as img_utils
import pose_utils.vis_utils as vis_utils
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from croco.models.croco import CroCoNet
import sys
sys.path.append("croco")

#ckpt = torch.load('/home/imw-mmi/PycharmProjects/ZS6D/pretrained_models/CroCo.pth')
#model = CroCoNet(**ckpt.get('croco_kwargs', {}))

# Loading the config file:
with open(os.path.join("./zs6d_configs/bop_eval_configs/cfg_ycbv_inference_bop.json"), "r") as f:
    config = json.load(f)

# Instantiating the pose estimator:
# This involves handing over the path to the templates_gt file and the corresponding object norm_factors.
pose_estimator = ZS6D(config['templates_gt_path'], config['norm_factor_path'], model_type='croco', subset_templates=5, max_crop_size=80)

# Loading a ground truth file to access segmentation masks to test zs6d:
with open(os.path.join(config['gt_path']), 'r') as f:
    data_gt = json.load(f)

img_id = '000048_1'
#img_id = '8'

for i in range(len(data_gt[img_id])):
    obj_number = i
    obj_id = data_gt[img_id][obj_number]['obj_id']
    cam_K = np.array(data_gt[img_id][obj_number]['cam_K']).reshape((3, 3))
    bbox = data_gt[img_id][obj_number]['bbox_visib']

    img_path = os.path.join(config['dataset_path'], data_gt[img_id][obj_number]['img_name'].split("./")[-1])

    img = Image.open('/home/imw/PycharmProjects/ZS6D/test/000001.png')

    mask = data_gt[img_id][obj_number]['mask_sam']
    mask = img_utils.rle_to_mask(mask)
    mask = mask.astype(np.uint8)

    start_time = time.time()

    # To estimate the objects Rotation R and translation t the input image, the object_id, a segmentation mask and camera matrix are necessary
    R_est, t_est = pose_estimator.get_pose(img, str(obj_id), mask, cam_K, bbox=None)

    end_time = time.time()

    out_img = vis_utils.draw_3D_bbox_on_image(np.array(img), R_est, t_est, cam_K,
                                              data_gt[img_id][obj_number]['model_info'], factor=1.0)

    plt.imshow(out_img)
    plt.show()
    print(f"Pose estimation time: {end_time - start_time}")
    print(f"R_est: {R_est}")
    print(f"t_est: {t_est}")
