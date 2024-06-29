from src.extractor import CroCoExtractor, pad_and_resize, ViTExtractor
import torch
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from croco.models.croco import CroCoNet
from croco.models.croco_downstream import CroCoDownstreamMonocularEncoder

# setting a seed so the model does not behave random
seed = 1  # found by checking the saliency map 33
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with torch.no_grad():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor_croco = CroCoExtractor(model_type='crocov1', stride=16, device=device)  # stride 16
    #extractor_croco = ViTExtractor(device=device)
    image_batch_croco1, image_pil_croco = extractor_croco.preprocess(
        '/home/stefan/Documents/ycbv_desc_enc_11_nobin_nocls/obj_6/000632.png', 224) #000248.png
    image_batch_croco1, image_pil_croco = extractor_croco.preprocess(
        '/home/stefan/Desktop/trash_dose.png', 224)  # 000248.png
    image_batch_croco2, image_pil_croco2 = extractor_croco.preprocess(
        '/home/stefan/Desktop/cut_dose.png', 224,)#000392.png
    image_pil_croco2.save('ddddddddd.png')


    def extract_descriptor(model, image):
        with torch.no_grad():
            features, _, _ = model._encode_image(image, do_mask=False, return_all_blocks=False)

        return features.mean(dim=1)  # Average over patches to get a single vector per image


    from torch import nn
    class PassThroughHead(nn.Module):
        def __init__(self):
            super(PassThroughHead, self).__init__()

        def setup(self, encoder):
            pass

        def forward(self, x, img_info):
            # Simply return the features without any modifications
            return x

    ckpt = torch.load('/home/stefan/PycharmProjects/ZS6D/pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth')
    head = PassThroughHead()
    model = CroCoDownstreamMonocularEncoder(**ckpt.get('croco_kwargs', {}), head=head, mask_ratio=0)
    desc1 = extract_descriptor(model, image_batch_croco1)
    desc2 = extract_descriptor(model, image_batch_croco2)

    # dot product of descriptors
    dot_product = torch.sum(desc1 * desc2, dim=1)

    # norm tensor
    norm_tensor1 = torch.norm(desc1, dim=1)
    norm_tensor2 = torch.norm(desc2, dim=1)

    # cosine similarity
    cosine_similarity= dot_product / (norm_tensor1 * norm_tensor2)
    mean = torch.mean(dot_product / (norm_tensor1 * norm_tensor2))

    print(cosine_similarity)
    print(mean)

