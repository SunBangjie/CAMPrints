import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
from torch import device # type: ignore
from torch.cuda import is_available # type: ignore
from torchvision import transforms # type: ignore
import pandas as pd
import dataloader.utils_image as util
import dataloader.adversarial as adv
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# define device
dev = device("cuda" if is_available() else "cpu")

# Define the EvaluationDataset class
class EvaluationDataset(Dataset):

    def __init__(self, csv_file, adv_level=[]):
        '''
        csv_file: path to the CSV file containing the image paths and device IDs
        adv_level: list of integers from 1 to 10, indicating the level of adversarial attacks
        '''

        # Load the CSV file from a specific path
        self.csv_file = csv_file
        self.dataset = pd.read_csv(self.csv_file, sep="\t")

        '''
        Columns from the CSV file:
            1. ref_img_paths: a list of reference image paths
            2. test_img_path: path to the test image
            3. ref_device_id: device ID of the reference image
            4. test_device_id: device ID of the test image
            5. is_query_in_reference: boolean value indicating if the test image is in the reference image
            6. is_same_device: boolean value indicating if the test image and the reference image are from the same device
        '''

        # assert adv_level must be a list of integers from 0 to 9
        assert all([level in range(11) for level in adv_level]), "adv_level must be a list of integers from 0 to 10"
        self.adv_level = adv_level
        
        # initialize the parameters
        self.crop_size = 512

        # constants
        self.IS_SAME_DEV = 0 # small embedding distance meaning same device
        self.IS_DIFF_DEV = 1 # large embedding distance meaning diff device

        
    def __len__(self):
        # get the size of the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # get the data at a specific index
        ref_img_paths = self.dataset.loc[idx, "ref_img_paths"].split(",")
        test_img_path = self.dataset.loc[idx, "test_img_path"]
        ref_device_id = int(self.dataset.loc[idx, "ref_device_id"])
        test_device_id = int(self.dataset.loc[idx, "test_device_id"])
        is_query_in_reference = bool(self.dataset.loc[idx, "is_query_in_reference"])
        is_same_device = bool(self.dataset.loc[idx, "is_same_device"])

        # Load the ref images from ref_img_paths
        ref_img_tensors = []
        for ref_img_path in ref_img_paths:
            ref_img_tensor = self._load_img(ref_img_path)
            ref_img_tensor = transforms.CenterCrop(self.crop_size)(ref_img_tensor) # Center crop the image
            ref_img_tensor = ref_img_tensor.to(dev) # Put the image in the device
            ref_img_tensors.append(ref_img_tensor)
        # Convert ref_img_tensors to a single tensor
        ref_img_tensors = torch.stack(ref_img_tensors)
        
        # Load the test image from test_img_path
        test_img_tensor = self._load_img(test_img_path)
        test_img_tensor = transforms.CenterCrop(self.crop_size)(test_img_tensor) # Center crop the image
        # Prepare for samples with adversarial attacks (1 - 4 are seen, 5 - 10 are unseen operations)
        if is_same_device:
            # we apply adv operations to the images from the same device to simulate attackers' efforts
            for level in self.adv_level:
                if level == 1:
                    test_img_tensor = adv.apply_instagram_filter(test_img_tensor)
                elif level == 2:
                    test_img_tensor = adv.apply_barrel_distortion(test_img_tensor)
                elif level == 3:
                    test_img_tensor = adv.apply_gaussian_noise(test_img_tensor)
                elif level == 4:
                    test_img_tensor = adv.random_crop_and_resize(test_img_tensor)
                elif level == 5:
                    # we always combine random rotation and random crop and resize to avoid black borders
                    test_img_tensor = adv.apply_random_rotation(test_img_tensor)
                    test_img_tensor = adv.random_crop_and_resize(test_img_tensor, 400)
                elif level == 6:
                    # we also combine random perspective transform and random crop and resize to avoid black borders
                    test_img_tensor = adv.apply_random_perspective_transform(test_img_tensor)
                    test_img_tensor = adv.random_crop_and_resize(test_img_tensor, 400)
                elif level == 7:
                    test_img_tensor = adv.apply_random_iphone_filters(test_img_tensor)
                elif level == 8:
                    test_img_tensor = adv.apply_random_gaussian_blur(test_img_tensor)
                elif level == 9:
                    test_img_tensor = adv.apply_random_median_filtering(test_img_tensor)
                elif level == 10:
                    test_img_tensor = adv.compress_jpeg(test_img_tensor, quality=80)
        test_img_tensor = test_img_tensor.to(dev) # Make sure all images are in the same device

        # convert the rest of the data to tensors
        ref_device_id = torch.tensor(ref_device_id, dtype=torch.long)
        test_device_id = torch.tensor(test_device_id, dtype=torch.long)
        label = torch.tensor(self.IS_SAME_DEV if is_same_device else self.IS_DIFF_DEV, dtype=torch.long)
        is_query_in_reference = torch.tensor(is_query_in_reference, dtype=torch.bool)
        
        return {
                "ref_imgs": ref_img_tensors,
                "ref_img_paths": ref_img_paths,
                "ref_device_id": ref_device_id,
                "test_img": test_img_tensor,
                "test_device_id": test_device_id,
                "test_label": label,
                "is_test_in_ref": is_query_in_reference
            }
    
    def _load_img(self, img_path):
        img = util.read_img(img_path)
        return util.uint2tensor3(img)

def load_img(img_path):
    img = util.read_img(img_path)
    return util.uint2tensor3(img)