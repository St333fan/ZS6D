from src.extractor import CroCoExtractor, pad_and_resize, ViTExtractor
import torch
import numpy as np
import random
import matplotlib
# Set the backend for matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt

# setting a seed so the model does not behave random
seed = 3# found by checking the saliency map 33
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
        '/home/stefan/Desktop/worst_dose.png', 224)#, mask=False) #000248.png
    #image_batch_croco1, image_pil_croco = extractor_croco.preprocess(
      #  '/home/stefan/PycharmProjects/ZS6D/test/000392.png', 224, mask=False)  # 000248.png
    image_batch_croco2, image_pil_croco2 = extractor_croco.preprocess(
        '/home/stefan/Desktop/cut_dose.png', 224)#, mask=False)#000392.png
    # Remove the batch dimension and move channels to the end
    image_array = image_batch_croco1.squeeze(0).permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()

    strings = ["key", "value", "query", 'token', 'attn']

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

            # Save the channel as a grayscale image
            plt.imsave('descriptor1.png', descriptors1_2d)
            plt.imsave('descriptor2.png', descriptors2_2d)

            # cosine similarity
            cosine_similarity.append(dot_product / (norm_tensor1 * norm_tensor2))
            import torch.nn.functional as F

            # mean cosine similarity
            mean_cosine_similarity.append(torch.mean(dot_product / (norm_tensor1 * norm_tensor2)))

        # Create a range for the x-axis (from 0 to 10)
        x_values = range(12)
        print(mean_cosine_similarity)
        # Create the line plot
        plt.plot(x_values, mean_cosine_similarity, label=facet.capitalize())

        # Add labels and title
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Line Plot of Values')

    # Save the plot as a PNG image
    plt.legend()
    plt.savefig('line_plot3.png')
