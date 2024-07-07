from src.extractor import CroCoExtractor, pad_and_resize, ViTExtractor
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import transforms
from collections import defaultdict
import random
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import torch

def replace_background(input_image, bg_color=(0, 255, 0)):
    # Convert the image to RGBA if it's not already
    img = input_image.convert('RGBA')

    # Convert the image to numpy array
    img_array = np.array(img)

    # Create a mask for black pixels (ignoring alpha channel)
    mask = (img_array[:, :, :3] == [0, 0, 0]).all(axis=2)

    # Create a background with the specified color and alpha
    bg = np.full(img_array.shape, bg_color + (255,), dtype=np.uint8)

    # Replace black pixels with the new background color, preserving original alpha
    result_array = img_array.copy()
    result_array[mask] = bg[mask]

    # Convert back to PIL Image and return
    return Image.fromarray(result_array)

# setting a seed so the model does not behave random
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with torch.no_grad():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    strings = ['key', 'attn']
    layer = 2

    # path to all objects
    folder_path = '/home/stefan/Documents/ycbv_desc_enc_11_nobin_nocls/obj_6'
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    results = defaultdict(list)
    extractor_croco = CroCoExtractor(model_type='crocov1', stride=16, device=device)

    # segmemented object
    image_batch_croco2, image_pil_croco2 = extractor_croco.preprocess(
        '/home/stefan/PycharmProjects/ZS6D/img_crop_647.png', 224)

    # Loop over the .png files
    for png_file in png_files:
        file_path = folder_path + '/' + png_file

        image_batch_croco1, image_pil_croco = extractor_croco.preprocess(
            image_path=file_path, load_size=224)

        if False:  # replace background test or apply blur (sometimes makes it better)
            #image_pil_croco = replace_background(image_pil_croco)
            from PIL import Image, ImageFilter
            # Apply Gaussian blur
            blurred = image_pil_croco.filter(ImageFilter.GaussianBlur(5))
            # Increase brightness
            image_pil_croco = Image.eval(blurred, lambda x: min(x * 1.2, 255))

            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1, 3, 1, 1).to(device, non_blocking=True)
            imagenet_std = [0.229, 0.224, 0.225]
            imagenet_std_tensor = torch.tensor(imagenet_std).view(1, 3, 1, 1).to(device, non_blocking=True)
            trfs = Compose([ToTensor(), Normalize(mean=imagenet_mean, std=imagenet_std), transforms.Resize((224, 224))])
            image_batch_croco1 = trfs(image_pil_croco.convert('RGB')).to(device, non_blocking=True).unsqueeze(0)


        var_name = png_file[:-4]
        # assign each string to the respective variable
        for i, facet in enumerate(strings):
            cosine_similarity = []
            mean_cosine_similarity = []

            descriptors1 = extractor_croco.extract_descriptors(image_batch_croco1.to(device), layer=layer, facet=facet,
                                                               bin=False,
                                                               include_cls=True)
            descriptors1_2d = descriptors1.cpu().squeeze(0).squeeze(0)
            descriptors2 = extractor_croco.extract_descriptors(image_batch_croco2.to(device), layer=layer, facet=facet,
                                                               bin=False,
                                                               include_cls=True)
            descriptors2_2d = descriptors2.cpu().squeeze(0).squeeze(0)

            # dot product of descriptors
            dot_product = torch.sum(descriptors1_2d * descriptors2_2d, dim=1)

            # norm tensor
            norm_tensor1 = torch.norm(descriptors1_2d, dim=1)
            norm_tensor2 = torch.norm(descriptors2_2d, dim=1)

            # cosine similarity
            cosine_similarity.append(dot_product / (norm_tensor1 * norm_tensor2))

            if facet == 'key':
                value1 = torch.mean(dot_product / (norm_tensor1 * norm_tensor2))
            else:
                value2 = torch.mean(dot_product / (norm_tensor1 * norm_tensor2))
        results[var_name] = [value1, value2]

    # Find the top ten highest values for each calculation
    top_ten_value1 = sorted(results.items(), key=lambda x: x[1][0], reverse=True)[:10]
    top_ten_value2 = sorted(results.items(), key=lambda x: x[1][1], reverse=True)[:10]

    # Print the results
    print("Top 10 highest values for calculation 1:")
    for name, values in top_ten_value1:
        print(f"{name}: {values[0]}")

    print("\nTop 10 highest values for calculation 2:")
    for name, values in top_ten_value2:
        print(f"{name}: {values[1]}")


