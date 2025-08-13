import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim
from scipy import ndimage


def get_similarity_score(im1, im2, metric='ssim'):
    if metric == 'ssim':
        score = ssim(im1, im2, win_size=3, multichannel=True, data_range=im1.max()-im1.min(), gaussian_weights=True)
    elif metric == 'psnr':
        mse = np.mean((im1 - im2) ** 2)
        if mse == 0:  # MSE is zero means no noise is present in the images
            return float('inf')
        score = 20 * np.log10(255.0 / np.sqrt(mse))  # Assuming 8-bit images
    elif metric == 'mse':
        score = np.mean((im1 - im2) ** 2)
    elif metric == 'ms_ssim':
        img1 = torch.tensor(im1.copy()).permute(2, 0, 1).unsqueeze(0).float()  # Shape: (1, 3, H, W)
        img2 = torch.tensor(im2.copy()).permute(2, 0, 1).unsqueeze(0).float()
        ms_ssim_value = ms_ssim(img1, img2, data_range=1.0)
        score = ms_ssim_value.item()
    elif metric == 'fsim':
        score = fsim(im1.copy(), im2.copy())
    return score

# Helper function to compute gradient magnitude
def gradient_magnitude(image):
    dx = ndimage.sobel(image, 0)  # horizontal gradient
    dy = ndimage.sobel(image, 1)  # vertical gradient
    magnitude = np.hypot(dx, dy)  # magnitude
    return magnitude

# Phase congruency computation (simplified)
def phase_congruency(image):
    # Using Sobel gradients as a proxy for phase congruency
    Gx = ndimage.sobel(image, axis=0)
    Gy = ndimage.sobel(image, axis=1)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    return magnitude / (np.max(magnitude) + 1e-8)

# Function to compute FSIM between two images
def fsim(imageA, imageB):
    if len(imageA.shape) == 3:  # If RGB, convert to grayscale
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
    # Normalize the images to [0, 1]
    imageA = imageA.astype(np.float64) / 255.0
    imageB = imageB.astype(np.float64) / 255.0

    # Compute phase congruency for both images
    PC_A = phase_congruency(imageA)
    PC_B = phase_congruency(imageB)

    # Compute gradient magnitude for both images
    GM_A = gradient_magnitude(imageA)
    GM_B = gradient_magnitude(imageB)

    # Similarity based on phase congruency
    S_pc = (2 * PC_A * PC_B + 1e-8) / (PC_A**2 + PC_B**2 + 1e-8)

    # Similarity based on gradient magnitude
    S_g = (2 * GM_A * GM_B + 1e-8) / (GM_A**2 + GM_B**2 + 1e-8)

    # Combine both similarities
    FSIM_value = np.sum(S_pc * S_g) / np.sum(S_g)
    return FSIM_value
