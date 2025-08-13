import torch
import torchvision.transforms.functional as F
from torch import device
from torch.cuda import is_available
import torchvision.transforms as T
import cv2
import numpy as np
import random
from PIL import Image
import io
from similarity_metrics import get_similarity_score

device = device("cuda" if is_available() else "cpu")

def adversarial_attack(img_tensor, attack_type, attack_param):
    """
    Apply an adversarial attack to the image tensor.
    
    Args:
    - image_tensor (torch.Tensor): The input image tensor with shape (C, H, W).
    - attack_type (int): The type of adversarial attack to apply. Options are 1, 2, 3, 4.
    
    Returns:
    - torch.Tensor: The image tensor with the adversarial attack applied.
    """
    if attack_type == 1:
        return apply_instagram_filter(img_tensor, filter_type=attack_param)
    elif attack_type == 2:
        return apply_barrel_distortion(img_tensor)
    elif attack_type == 3:
        return apply_gaussian_noise(img_tensor, mean=0.0, std=attack_param)
    elif attack_type == 4:
        return random_crop_and_resize(img_tensor)
    elif attack_type == 5:
        # we always combine random rotation and random crop and resize to avoid black borders
        img_tensor = apply_random_rotation(img_tensor, max_rotation_angle=attack_param)
        return random_crop_and_resize(img_tensor, 400)
    elif attack_type == 6:
        # we always combine random rotation and random crop and resize to avoid black borders
        img_tensor = apply_random_perspective_transform(img_tensor)
        return random_crop_and_resize(img_tensor, 400)
    elif attack_type == 7:
        return apply_random_iphone_filters(img_tensor)
    elif attack_type == 8:
        return apply_random_gaussian_blur(img_tensor, max_kernel_size=attack_param)
    elif attack_type == 9:
        return apply_random_median_filtering(img_tensor, max_kernel_size=attack_param)
    elif attack_type == 10:
        return compress_jpeg(img_tensor, quality=attack_param)
    else:
        raise ValueError("attack_type must be in [1, 9]")

def apply_adv_attacks(image_tensor, attack_types, attack_params):
    original_image_tensor = image_tensor.clone()
    for i, attack_type in enumerate(attack_types):
        attack_param = attack_params[i]
        image_tensor = adversarial_attack(image_tensor, attack_type, attack_param)
    # compute the similarity score between the original and the adversarial image
    ref_img = original_image_tensor.permute(1, 2, 0).cpu().numpy()
    test_img = image_tensor.permute(1, 2, 0).cpu().numpy()
    fsim_score = get_similarity_score(ref_img, test_img, metric='fsim')
    sim_scores_after_processing = fsim_score
    return image_tensor, sim_scores_after_processing


def apply_instagram_filter(image_tensor, filter_type='random'):
    """
    Apply an Instagram-like filter to the image tensor.
    
    Args:
    - image_tensor (torch.Tensor): The input image tensor with shape (C, H, W).
    - filter_type (str): The type of Instagram filter to apply. Options are 'clarendon', 'gingham', 'moon'.
    
    Returns:
    - torch.Tensor: The image tensor with the Instagram filter applied.
    """

    # Move image_tensor to the device
    image_tensor = image_tensor.to(device)

    # Randomly select filter_type
    if filter_type == 'random':
        filter_type = torch.randint(3, (1,)).item()
        filter_type = ['clarendon', 'gingham', 'moon'][filter_type]

    if filter_type == 'clarendon':
        # Increase contrast and saturation
        image_tensor = F.adjust_contrast(image_tensor, 1.5)
        image_tensor = F.adjust_saturation(image_tensor, 1.5)
    elif filter_type == 'gingham':
        # Reduce contrast and add a slight tint
        image_tensor = F.adjust_contrast(image_tensor, 0.8)
        image_tensor = F.adjust_brightness(image_tensor, 1.1)
        image_tensor = F.adjust_saturation(image_tensor, 0.9)
        image_tensor = image_tensor + torch.tensor([0.1, 0.1, 0.1], device=device).view(3, 1, 1)  # Add a slight tint
    elif filter_type == 'moon':
        # Convert to grayscale and increase contrast
        image_tensor = F.rgb_to_grayscale(image_tensor, num_output_channels=3)
        image_tensor = F.adjust_contrast(image_tensor, 1.3)
    else:
        raise ValueError("filter_type must be 'clarendon', 'gingham', or 'moon'")
    
    return torch.clamp(image_tensor, 0, 1)


def apply_gaussian_noise(image_tensor, mean=0.0, std=0.01):
    """
    Apply random Gaussian noise to the image tensor.
    
    Args:
    - image_tensor (torch.Tensor): The input image tensor with shape (C, H, W).
    - mean (float): The mean of the Gaussian noise.
    - std (float): The standard deviation of the Gaussian noise.
    
    Returns:
    - torch.Tensor: The image tensor with Gaussian noise applied.
    """

    # Move image_tensor to the device
    image_tensor = image_tensor.to(device)

    noise = torch.randn(image_tensor.size(), device=device) * std + mean
    noisy_image_tensor = image_tensor + noise
    return torch.clamp(noisy_image_tensor, 0, 1)  # Ensure the values are within [0, 1]


def apply_barrel_distortion(image_tensor):
    """
    Apply barrel distortion to the image tensor.
    
    Args:
    - image_tensor (torch.Tensor): The input image tensor with shape (C, H, W).
    - distortion_coefficient (float): The coefficient for barrel distortion. Higher values mean more distortion.
    
    Returns:
    - torch.Tensor: The image tensor with barrel distortion applied.
    """
    # Generate a random distortion strength
    min_distortion_strength = -0.1
    max_distortion_strength = -0.05
    distortion_strength = torch.rand(1).item() * (max_distortion_strength - min_distortion_strength) + min_distortion_strength

    C, H, W = image_tensor.shape
    
    # Create meshgrid with default 'xy' indexing
    x, y = torch.meshgrid(torch.arange(W, device=image_tensor.device), 
                          torch.arange(H, device=image_tensor.device))
    
    # Transpose x and y to simulate 'ij' indexing
    x = x.t().float()
    y = y.t().float()

    # Normalize coordinates to [-1, 1]
    x = 2 * (x / (W - 1)) - 1
    y = 2 * (y / (H - 1)) - 1

    # Calculate radial distance from center
    r = torch.sqrt(x**2 + y**2)
    rd = r * (1 + distortion_strength * r**2)

    # Convert radial distances back to distorted x, y coordinates
    distorted_x = rd * x / (r + 1e-6)  # Avoid division by zero
    distorted_y = rd * y / (r + 1e-6)

    # Create grid for grid_sample
    grid = torch.stack([distorted_x, distorted_y], dim=-1).unsqueeze(0)

    # Apply the distortion using grid_sample
    distorted_image = torch.nn.functional.grid_sample(image_tensor.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return distorted_image.squeeze(0)


def random_crop_and_resize(image_tensor, crop_size=None):
    """
    Randomly crop and resize the image tensor.
    
    Args:
    - image_tensor (torch.Tensor): The input image tensor with shape (C, H, W).
    
    Returns:
    - torch.Tensor: The image tensor with random crop and resize applied.
    """
    # Move image_tensor to the device
    image_tensor = image_tensor.to(device)

    # Get the dimension of the image tensor
    _, H, W = image_tensor.shape

    if crop_size is None:
        # Randomly select crop size
        crop_size = torch.randint(H//2, H, (1,)).item()

    # Randomly select crop position
    x_offset = torch.randint(0, H - crop_size, (1,)).item()
    y_offset = torch.randint(0, W - crop_size, (1,)).item()

    # Crop the image
    cropped_image = F.crop(image_tensor, y_offset, x_offset, crop_size, crop_size)

    # Resize the cropped image to the original size
    resized_image = F.resize(cropped_image, (H, W))

    return resized_image


def apply_random_rotation(image_tensor, max_rotation_angle=5):
    """
    Apply random rotation to the image tensor.
    
    Args:
    - image_tensor (torch.Tensor): The input image tensor with shape (C, H, W).
    
    Returns:
    - torch.Tensor: The image tensor with random rotation applied.
    """
    # Move image_tensor to the device
    image_tensor = image_tensor.to(device)

    # Randomly select rotation angle
    rotation_angle = torch.randint(-max_rotation_angle, max_rotation_angle, (1,)).item()

    # Rotate the image
    rotated_image = F.rotate(image_tensor, rotation_angle)

    return rotated_image


# Define a function to apply random perspective transformation on a tensor
def apply_random_perspective_transform(image_tensor):
    random_scale = random.random() * 0.2  # Random scale between 0 and 0.2
    random_perspective_transform = T.RandomPerspective(distortion_scale=random_scale, p=1.0)
    return random_perspective_transform(image_tensor)


# Define a function to apply random iPhone-like filters
def apply_random_iphone_filters(image_tensor):
    # Randomly adjust brightness, contrast, saturation, and hue
    image_tensor = F.adjust_brightness(image_tensor, brightness_factor=torch.FloatTensor(1).uniform_(0.8, 1.2).item())
    image_tensor = F.adjust_contrast(image_tensor, contrast_factor=torch.FloatTensor(1).uniform_(0.8, 1.2).item())
    image_tensor = F.adjust_saturation(image_tensor, saturation_factor=torch.FloatTensor(1).uniform_(0.8, 1.3).item())
    image_tensor = F.adjust_hue(image_tensor, hue_factor=torch.FloatTensor(1).uniform_(-0.05, 0.05).item())
    
    # Optionally, apply a random warm or cool color tint
    tint_choice = torch.randint(0, 3, (1,)).item()  # 0 = no tint, 1 = warm tint, 2 = cool tint
    
    if tint_choice == 1:  # Warm filter
        # Add a reddish-yellow tint
        red_multiplier = torch.FloatTensor(1).uniform_(1.0, 1.2).item()
        image_tensor = image_tensor * (torch.tensor([red_multiplier, 1.0, 0.9]).to(image_tensor.device)).view(3, 1, 1)
    
    elif tint_choice == 2:  # Cool filter
        # Add a bluish tint
        blue_multiplier = torch.FloatTensor(1).uniform_(1.0, 1.2).item()
        image_tensor = image_tensor * (torch.tensor([0.9, 1.0, blue_multiplier]).to(image_tensor.device)).view(3, 1, 1)
    
    # Ensure values stay in [0, 1] range
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    return image_tensor


def apply_random_gaussian_blur(image_tensor, max_kernel_size=7):
    # Randomly decide the kernel size for the Gaussian blur (odd values)
    kernel_size = int(torch.randint(0, max_kernel_size//2, (1,)).item() * 2 + 3)  # Select from {3, 5, 7}
    
    # Apply Gaussian blur for noise removal
    blurred_image_tensor = F.gaussian_blur(image_tensor, kernel_size=[kernel_size, kernel_size])

    # Make sure blurred_imahe_tensor is in the range [0, 1]
    blurred_image_tensor = torch.clamp(blurred_image_tensor, 0, 1)
    
    return blurred_image_tensor


def apply_random_median_filtering(image_tensor, max_kernel_size=7):
    # Convert PyTorch tensor to a NumPy array (convert to [H, W, C] and scale to 0-255)
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255
    image_np = image_np.astype(np.uint8)  # Ensure the type is uint8 for OpenCV
    
    # Randomly select an odd kernel size for the median filter (3, 5, or 7)
    kernel_size = int(torch.randint(3, max_kernel_size+1, (1,)).item())
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd (median filter requires odd values)
    
    # Apply median filtering
    image_np_filtered = cv2.medianBlur(image_np, kernel_size)
    
    # Convert back to PyTorch tensor ([H, W, C] to [C, H, W] and normalize to [0, 1])
    image_tensor_filtered = torch.from_numpy(image_np_filtered).permute(2, 0, 1).float() / 255.0
    
    return image_tensor_filtered


def compress_jpeg(tensor, quality=90):
    # Convert the PyTorch tensor to a PIL image
    transform_to_pil = T.ToPILImage()
    pil_img = transform_to_pil(tensor)
    
    # Save the PIL image to a BytesIO buffer with JPEG compression
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    
    # Load the image back from the buffer
    buffer.seek(0)
    pil_img_compressed = Image.open(buffer)
    
    # Convert the compressed PIL image back to a PyTorch tensor
    transform_to_tensor = T.ToTensor()
    compressed_tensor = transform_to_tensor(pil_img_compressed)
    
    return compressed_tensor