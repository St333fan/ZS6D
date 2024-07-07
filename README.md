# ZS6D with added CroCo support and Cross Completion Match (CroCoM)


![dino_croco](./assets/Robot_Vision_CroCo_Dino.png)

Dino vs CroCo (Cross View Completion, self-supervised pre-trained ViT)

![cpipeline](./assets/croco_pipeline.png)

After coming to the conclusion that Dino is superior to CroCo in descriptor matching, a new pipeline Cross Completion Match (CroCoM) is proposed for
template matching building on the self-supervised training method. Cross view completion allows us to compare all
templates with each other to find the best match for the segmented to-be-found object in 6D by using the reconstruction task of CroCo.

## How to install Git for CroCo and CroCoM 
- Please go to the section "Overview of the original ZS6D-Dino Project" and follow the installation process from there.
- If you are on Ubuntu 22.04, there will be an error, to solve it go to https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris
- When Cv2 makes problems
```
sudo apt-get install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xkb1
```
- If your graphics card has less than 32 GByte VRAM create 32 GByte swap-memory, we tested it with 16GBVRAM and 32GBSWAP

## How to run ZS6D with CroCo 
- **General rule, you will run into several PATH issues! I mentioned the most important ones!**
- Download **CroCo.pth** and **CroCo_V2_ViTLarge_BaseDecoder.pth** from the original croco git and put them into [pretrained_models](pretrained_models)
- Change import statements in the croco (subgit), there should be 4, your IDE will mark it either way
- [prepare_templates_and_gt_croco.py](prepare_templates_and_gt_croco.py) prepares the croco descriptors, if you want to run it change the paths to your dataset in [cfg_template_gt_generation_ycbv_croco.json](zs6d_configs%2Ftemplate_gt_preparation_configs%2Fcfg_template_gt_generation_ycbv_croco.json)
- [bop_eval_configs](zs6d_configs%2Fbop_eval_configs) check all the paths in the **.json files**
- [test_zs6d_croco.py](test_zs6d_croco.py) is the test script, if you run it, it will probably find no pose because it seems that CroCo does not work in the ZS6D pipeline
- To test with **CroCo_V2_ViTLarge_BaseDecoder** just exchange it in the function call with **crocov2**
---
- evaluation is done with [evaluate_zs6d_bop_croco.py](evaluate_zs6d_bop_croco.py), check also the files in [bop_eval_configs](zs6d_configs%2Fbop_eval_configs)
- analyse the created evaluation data with [analyse_evaluated_zs6d_data.py](analyse_evaluated_zs6d_data.py)

## How to run CroCoM
- Try to run CroCo first, because it also has some **setup steps** which are needed for **CroCoM**, if you are Pro you can try to go straight to CroCoM and debug on the fly 
- In [pretrained_models](pretrained_models) is [crocom.py](pretrained_models%2Fcrocom.py) put it into the folder from the **original corco (subgit)** where **croco.py** is found
- Start [croco_match.py](croco_match.py) for testing on **single segmented objects**, by changing the paths in the **main** function
- [evaluate_zs6d_bop_crocom.py](evaluate_zs6d_bop_crocom.py) evaluates on **myset (small ybvc testset)**
- **There is currently no implementation for CroCoM in test_zs6d_crocom.py it does not exist!**
---
- Evaluation is done with [evaluate_zs6d_bop_crocom.py](evaluate_zs6d_bop_crocom.py), check also the files in [bop_eval_configs](zs6d_configs%2Fbop_eval_configs)
- Analyse the created evaluation data with [analyse_evaluated_zs6d_data.py](analyse_evaluated_zs6d_data.py)



## For what are the additional Scripts?
**No working paths are guaranteed! Personal testing Scripts!**

Testing a specific layer and token, to find the best matches of the templates to segmented object
- [test_croco_layer_against_all.py](test_croco_layer_against_all.py)


Testing all layers and tokens, on one template and one segmented object
- [test_croco_layers.py](test_croco_layers.py)


Testing the output of the CroCoDownstreamMonocularEncoder
- [test_croco_output.py](test_croco_output.py)






## Overview of the original ZS6D-Dino Project:

![pipeline](./assets/ZS6D_pipeline.png)

Note that this repo only deals with 6D pose estimation, you need segmentation masks as input. These can be obtained with supervised trained methods or zero-shot methods. For zero-shot we refer to [cnos](https://github.com/nv-nguyen/cnos).
<p align="center">
  <img src="./assets/overview.png" width="500" alt="teaser"/>
</p>
We demonstrate the effectiveness of deep features extracted from self-supervised, pre-trained Vision Transformer (ViT) for Zero-shot 6D pose estimation. For more detailed information check out the corresponding [[paper](https://arxiv.org/pdf/2309.11986.pdf)].

## Installation:
To setup the environment to run the code locally follow these steps:

```
conda env create -f environment.yml
conda activate zs6d
```

Otherwise, run the following commands:

```
conda create --name zs6d python=3.9
conda activate zs6d
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install tqdm==4.65.0
pip install timm==0.9.16
pip install matplotlib==3.8.3
pip install scikit-learn==1.4.1.post1
pip install opencv-python==4.9.0
pip install git+https://github.com/lucasb-eyer/pydensecrf.git@dd070546eda51e21ab772ee6f14807c7f5b1548b
pip install transforms3d==0.4.1
pip install pillow==9.4.0
pip install plyfile==1.0.3
pip install trimesh==4.1.4
pip install imageio==2.34.0
pip install pypng==0.20220715.0
pip install vispy==0.12.2
pip install pyopengl==3.1.1a1
pip install pyglet==2.0.10
pip install numba==0.59.0
pip install jupyter==1.0.0
```


### Docker setup:

### ROS integration:

## Template rendering:
To generate templates from a object model to perform inference, we refer to the [ZS6D_template_rendering](https://github.com/haberger/ZS6D_template_rendering) repository.

## Template preparation:

1. set up a config file for template preparation

```zs6d_configs/template_gt_preparation_configs/your_template_config.json```

2. run the preparation script with your config_file to generate your_template_gt_file.json and prepare the template descriptors and template uv maps

```python3 prepare_templates_and_gt.py --config_file zs6d_configs/template_gt_preparation_configs/your_template_config.json```


## Inference:

1. download the pretrained croco and put it into the pretrained_models folder

```wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth -P pretrained_models/```

2. After setting up your_template_config.json you can instantiate your ZS6D module and perform inference. An example is provided in:

```test_zs6d.ipynb```


## Evaluation on BOP Datasets:

1. set up a config file for BOP evaluation

```zs6d_configs/bop_eval_configs/your_eval_config.json```

2. Create a ground truth file for testing, the files for BOP'19-23 test images are provided for lmo, tless and ycbv. For example for lmo:

```gts/test_gts/lmo_bop_test_gt_sam.json```

Additionally, you have to download the corresponding [BOP test images](https://bop.felk.cvut.cz/datasets/#LM-O). If you want to test another dataset as the provided, you have to generate a ground truth file with the following structure:

```json
{
  "object_id": [
    {
      "scene_id": "00001", 
      "img_name": "relative_path_to_image/image_name.png", 
      "obj_id": "..", 
      "bbox_obj": [], 
      "cam_t_m2c": [], 
      "cam_R_m2c": [], 
      "cam_K":[],
      "mask_sam": [] // mask in RLE encoding
    }
    ,...
  ]
}
```

3. run the evaluation script with your_eval_config.json

```python3 prepare_templates_and_gt.py --config_file zs6d_configs/template_gt_preparation_configs/your_eval_config.json```


## Acknowledgements
This project is built upon [dino-vit-features](https://github.com/ShirAmir/dino-vit-features), which performed a very comprehensive study about features of self-supervised pretrained Vision Transformers and their applications, including local correspondence matching. Here is a link to their [paper](https://arxiv.org/abs/2112.05814). We thank the authors for their great work and repo.

## Citation
If you found this repository useful please consider starring ⭐ and citing :

```
@article{ausserlechner2023zs6d,
  title={ZS6D: Zero-shot 6D Object Pose Estimation using Vision Transformers},
  author={Ausserlechner, Philipp and Haberger, David and Thalhammer, Stefan and Weibel, Jean-Baptiste and Vincze, Markus},
  journal={arXiv preprint arXiv:2309.11986},
  year={2023}
}
```
