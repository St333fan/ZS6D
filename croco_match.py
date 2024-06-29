import os
import torch
import time
from croco.models.crocom import CroCoNet
from PIL import Image
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.color import rgb2gray
from skimage.feature import canny
from heapq import heappush, heappushpop
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def create_custom_mask(mask_array):
    """
    Creates a custom mask based on a 2D array of 0s and 1s.

    Args:
    mask_array: 2D numpy array or list of lists where 0 is visible and 1 is masked

    Returns:
    mask: torch.Tensor, boolean mask of shape (1, num_patches)
    """
    mask_array = np.array(mask_array, dtype=bool)
    num_patches = mask_array.size
    mask = torch.from_numpy(mask_array.flatten()).unsqueeze(0)
    return mask

def visualize_mask(mask, grid_size):
    """
    Visualizes the mask.

    Args:
    mask: torch.Tensor, boolean mask of shape (1, num_patches)
    grid_size: int, size of the grid (height/width of the mask array)
    """
    mask_np = mask.reshape(grid_size, grid_size).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(mask_np, cmap='gray_r')
    plt.title('Custom Mask Visualization')
    plt.axis('off')
    plt.show()

def calculate_advanced_similarity(img1, img2):
    """
        Compute several similarity metrics

        Args:
        img1: torch.Tensor
        img2: torch.Tensor

        Returns:
        tuple: A tuple containing:
            - combined_similarity (float): An overall similarity score.
            - dict: A dictionary containing the following similarity metrics:
                - 'ssim' (float): Structural Similarity Index
                - 'color_similarity' (float): Color similarity metric
                - 'edge_similarity' (float): Edge similarity metric
                - 'inverse_mse' (float): Inverse of Mean Squared Error
                - 'warp_similarity' (float): Warp similarity metric scaled by 10
    """
    img1 = (img1.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img2 = (img2.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Grayscale for luminance SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Calculate SSIM for luminance
    ssim_luminance, _ = ssim(gray1, gray2, full=True, data_range=255)

    # Calculate SSIM for each color channel
    ssim_red, _ = ssim(img1[:,:,0], img2[:,:,0], full=True, data_range=255)
    ssim_green, _ = ssim(img1[:,:,1], img2[:,:,1], full=True, data_range=255)
    ssim_blue, _ = ssim(img1[:,:,2], img2[:,:,2], full=True, data_range=255)

    # Calculate mean SSIM across color channels
    ssim_color = (ssim_red + ssim_green + ssim_blue) / 3

    # Combine luminance and color SSIM (you can adjust weights)
    ssim_value = 0.5 * ssim_luminance + 0.5 * ssim_color

    # 2. Color Histogram Similarity
    def color_histogram_similarity(img1, img2):
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    color_sim = color_histogram_similarity(img1, img2)

    # 3. Edge Similarity
    def edge_similarity(img1, img2):
        edges1 = canny(rgb2gray(img1))
        edges2 = canny(rgb2gray(img2))
        return np.mean(edges1 == edges2)

    edge_sim = edge_similarity(img1, img2)

    # 4. Mean Squared Error (inverse, as lower is better)
    mse = 1 / (1 + np.mean((img1 - img2) ** 2))

    # 5. Warping Detection using ORB
    def warping_similarity(img1, img2):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # Use BFMatcher to find the best matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate the ratio of good matches to total matches
        good_matches = [m for m in matches if m.distance < 50]  # Adjust threshold as needed
        match_ratio = len(good_matches) / max(len(kp1), len(kp2))

        return match_ratio

    warp_sim = warping_similarity(img1, img2)
    def calculate_sharpness(image):

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Compute the Laplacian of the image and then return the focus measure
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    print(calculate_sharpness(img2))

    # Combine all metrics adjust wights as needed
    combined_similarity = (ssim_value + color_sim + edge_sim + mse + 10 * warp_sim) / 5

    return combined_similarity, {
        'ssim': ssim_value,
        'color_similarity': color_sim,
        'edge_similarity': edge_sim,
        'inverse_mse': mse,
        'warp_similarity': warp_sim*10
    }

def process_image(model, image_path, ref_image, device, trfs, imagenet_mean_tensor, imagenet_std_tensor, mask_array):
    """
    Process an image using a given model and compare it to a reference image.

    This function loads an image, applies transformations, runs it through a model along with a reference image,
    decodes the output, and calculates the similarity between the decoded image and the input image.

    Parameters:
    model (torch.nn.Module): The neural network model to use for processing.
    image_path (str): Path to the image file to be processed.
    ref_image (torch.Tensor): The reference image tensor.
    device (torch.device): The device (CPU or GPU) to run the computations on.
    trfs (torchvision.transforms.Compose): Composition of image transformations to apply.
    imagenet_mean_tensor (torch.Tensor): Mean tensor for ImageNet normalization.
    imagenet_std_tensor (torch.Tensor): Standard deviation tensor for ImageNet normalization.
    mask_array (numpy.ndarray): Array used to create a custom mask.

    Returns:
    torch.Tensor: The decoded image tensor.
    """
    image1 = ref_image
    image2 = trfs(Image.open(image_path).convert('RGB')).to(device, non_blocking=True).unsqueeze(0)

    custom_mask = create_custom_mask(mask_array)

    with torch.inference_mode():
        out, mask, target = model(image1, image2, custom_mask=custom_mask)

    patchified = model.patchify(image1)
    mean = patchified.mean(dim=-1, keepdim=True)
    var = patchified.var(dim=-1, keepdim=True)
    decoded_image = model.unpatchify(out * (var + 1.e-6)**.5 + mean)
    decoded_image = decoded_image * imagenet_std_tensor + imagenet_mean_tensor

    # Calculate similarity between image2 and decoded_image
    similarity = calculate_advanced_similarity(image2 * imagenet_std_tensor + imagenet_mean_tensor,
                                decoded_image)
    print(f"Similarity between reference image and decoded image for {os.path.basename(image_path)}: {similarity}")

    return decoded_image

def process(ref_image_path, ckpt_path, output_folder, assets_folder, mask_array):
    """
        Process a set of images using a reference image and a pre-trained model.

        This function sets up the environment, loads a model and a reference image,
        and then processes all images in a specified folder, saving the decoded results.

        Parameters:
        ref_image_path (str): Path to the reference image file.
        ckpt_path (str): Path to the checkpoint file containing the pre-trained model.
        output_folder (str): Path to the folder where decoded images will be saved.
        assets_folder (str): Path to the folder containing images to be processed.

        Returns:
        None
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1,3,1,1).to(device, non_blocking=True)
    imagenet_std = [0.229, 0.224, 0.225]
    imagenet_std_tensor = torch.tensor(imagenet_std).view(1,3,1,1).to(device, non_blocking=True)
    trfs = Compose([ToTensor(), Normalize(mean=imagenet_mean, std=imagenet_std),transforms.Resize((224, 224))])

    # Load the reference image
    ref_image = trfs(Image.open(ref_image_path).convert('RGB')).to(device, non_blocking=True).unsqueeze(0)

    # load model
    ckpt = torch.load(ckpt_path, 'cpu')
    model = CroCoNet(**ckpt.get('croco_kwargs', {}), mask_ratio=0.9).to(device)
    model.eval()
    model.load_state_dict(ckpt['model'], strict=True)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(assets_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(assets_folder, filename)
            decoded_image = process_image(model, image_path, ref_image, device, trfs, imagenet_mean_tensor,
                                          imagenet_std_tensor, mask_array)

            # Save the decoded image
            output_path = os.path.join(output_folder, f'decoded_{filename}')
            torchvision.utils.save_image(decoded_image, output_path)
            print(f'Decoded image saved: {output_path}')

def find_match(ref_image_path, decoded_images_dir, mask_array):
    """
        Match a reference image with several decoded images

        Args:
        ref_image_path: String, path to the reference image file
        decoded_images_dir: String, path to the decoded image files
        mask_array: np.Array, used mask

        Returns:
        String: Name of best matched image
    """
    def expand_mask(mask_array, patch_size):
        # Convert mask array to numpy array
        mask = np.array(mask_array)

        # Expand mask to full image size
        expanded_mask = np.repeat(np.repeat(mask, patch_size, axis=0), patch_size, axis=1)

        return expanded_mask

    def apply_mask_to_image(image, mask):
        # Ensure image and mask have the same size
        assert image.shape[:2] == mask.shape, "Image and mask must have the same dimensions"

        # Apply mask
        masked_image = image.copy()
        masked_image[mask == 0] = 0  # Set pixels to black where mask is 0

        return masked_image

    # Expand the mask
    expanded_mask = expand_mask(mask_array, 16)
    def measure_quality(img1, img2):
        # Ensure images are the same size
        h, w = img2.shape[:2]
        img1 = cv2.resize(img1, (w, h))

        # Convert images to grayscale for SSIM
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        ssim_value = ssim(gray1, gray2,
                          data_range=gray1.max() - gray1.min(),
                          win_size=min(7, min(gray1.shape) - 1))  # Ensure win_size is odd and smaller than image

        # Calculate MSE
        mse = np.mean((img1 - img2) ** 2)

        return ssim_value, mse

    # Load images
    img1 = cv2.imread(ref_image_path)

    top_10 = []

    # Process all images in the directory
    for filename in os.listdir(decoded_images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(decoded_images_dir, filename)
            img2 = cv2.imread(img_path)
            print(f"Processing: {filename}")

            # Apply mask
            img2 = apply_mask_to_image(img2, expanded_mask)

            # Measure quality
            ssim_value, mse = measure_quality(img1, img2)

            print(f"SSIM: {ssim_value:.4f}")
            print(f"MSE: {mse:.4f}")

            # Use a max heap to keep track of the top 10 matches
            if len(top_10) < 10:
                heappush(top_10, (-mse, filename))
            elif -mse > top_10[0][0]:
                heappushpop(top_10, (-mse, filename))

    # Sort the results by MSE (ascending order)
    top_10.sort(key=lambda x: -x[0])

    # Print results
    print("\nTop 10 matches:")
    for i, (neg_mse, filename) in enumerate(top_10, 1):
        print(f"{i}. {filename} (MSE: {-neg_mse:.4f})")

    # Return the name of the best match
    return top_10[0][1]

def main():
    start_time = time.time()
    # Alternative masks
    if False:
        image_size = 224
        patch_size = 16

        custom_mask = create_custom_mask(image_size, patch_size)
        mask_array = [
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
        ]

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

    process('/home/stefan/PycharmProjects/ZS6D/test/drill/3.png',
                     '/home/stefan/PycharmProjects/ZS6D/pretrained_models/CroCo.pth',#_V2_ViTLarge_BaseDecoder
                     '/home/stefan/PycharmProjects/ZS6D/assets_match/decoded_images',
                    '/home/stefan/PycharmProjects/ZS6D/assets_match', mask_array)

    best_match = find_match('/home/stefan/PycharmProjects/ZS6D/test/drill/3.png',
                             '/home/stefan/PycharmProjects/ZS6D/assets_match/decoded_images', mask_array)

    print(f"The image with the lowest MSE is: {best_match}")

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.2f} seconds")

if __name__=="__main__":
    main()
