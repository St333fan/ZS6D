from src.extractor import CroCoExtractor, pad_and_resize
import torch
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# setting a seed so the model does not behave random
seed = 33  # found by checking the saliency map
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with torch.no_grad():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor_croco = CroCoExtractor(model_type='croco', stride=16, device=device)  # stride 16

    image_batch_croco1, image_pil_croco = extractor_croco.preprocess(
        '/home/stefan/PycharmProjects/ZS6D/test/000392.png', 224) #000248.png
    image_batch_croco2, image_pil_croco2 = extractor_croco.preprocess(
        '/home/stefan/PycharmProjects/ZS6D/test/maskcut.png', 224)#000392.png

    image_batch_croco1 = pad_and_resize(image_batch_croco1)
    image_batch_croco2 = pad_and_resize(image_batch_croco2)

    strings = ["key", "value", "query"]

    # Loop over the list and assign each string to the respective variable
    for i, facet in enumerate(strings):
        cosine_similarity = []
        mean_cosine_similarity = []
        for layer in range(12):
            descriptors1 = extractor_croco.extract_descriptors(image_batch_croco1.to(device), layer=layer, facet=facet, bin=False,
                                                               include_cls=True)
            descriptors1_2d = descriptors1.cpu().squeeze(0).squeeze(0)
            descriptors2 = extractor_croco.extract_descriptors(image_batch_croco2.to(device), layer=layer, facet=facet, bin=False,
                                                               include_cls=True)
            descriptors2_2d = descriptors2.cpu().squeeze(0).squeeze(0)

            # dot product of descriptors
            dot_product = torch.sum(descriptors1_2d * descriptors2_2d, dim=1)

            # norm tensor
            norm_tensor1 = torch.norm(descriptors1_2d, dim=1)
            norm_tensor2 = torch.norm(descriptors2_2d, dim=1)

            # cosine similarity
            cosine_similarity.append(dot_product / (norm_tensor1 * norm_tensor2))

            # mean cosine similarity
            mean_cosine_similarity.append(torch.mean(dot_product / (norm_tensor1 * norm_tensor2)))

        # Create a range for the x-axis (from 0 to 10)
        x_values = range(12)

        # Create the line plot
        plt.plot(x_values, mean_cosine_similarity, label=facet.capitalize())

        # Add labels and title
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Line Plot of Values')

    # Save the plot as a PNG image
    plt.legend()
    plt.savefig('line_plot1.png')

