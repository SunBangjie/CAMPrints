import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
from torch import device # type: ignore
from torch.cuda import is_available # type: ignore
from torchvision import transforms # type: ignore
import pandas as pd
import dataloader.utils_image as util
import dataloader.adversarial as adv
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# define device
dev = device("cuda" if is_available() else "cpu")

# Define the CameraDataset class
class CameraDataset(Dataset):

    def __init__(self, train=True, adv_level=0, crop_size=512, selected_device=None, background_only=False):
        self.csv_file = '/home/bangjie/CamHash_v2/Data_Processing/SOCRatES_Selected_Device_Info_Pairs_New.csv'
        df = pd.read_csv(self.csv_file)

        assert adv_level in [0, 1, 2, 3, 4], "adv_level must be 0, 1, 2, 3, or 4"
        self.adv_level = adv_level

        self.crop_size = crop_size

        '''
        Columns from the CSV file:
            - 0: anc_img_path
            - 1: pos_img_path
            - 2: neg_img_path
            - 3: anc_device_id
            - 4: pos_device_id
            - 5: neg_device_id
            - 6: anc_FG/BG
            - 7: neg_FG/BG
            - 8: train/test
        '''

        self.labels = list(set(list(df[df.columns[3]].unique()) + list(df[df.columns[5]].unique())))
        
        if train:
            self.camera_images = df[df[df.columns[8]]==0]
        else:
            self.camera_images = df[df[df.columns[8]]==1]
        
        if selected_device is not None:
            # selected_device is a list of device ids
            self.camera_images = self.camera_images[self.camera_images[df.columns[3]].isin(selected_device)]
        
        if background_only:
            self.camera_images = self.camera_images[self.camera_images[df.columns[6]]==0]

    def __len__(self):
        return len(self.camera_images)

    def __getitem__(self, idx):
        # Load images
        anc_img_tensor = self._load_img(self.camera_images.iloc[idx, 0])
        pos_img_tensor = self._load_img(self.camera_images.iloc[idx, 1])
        neg_img_tensor = self._load_img(self.camera_images.iloc[idx, 2])

        # Center crop the image
        anc_img_tensor = transforms.CenterCrop(self.crop_size)(anc_img_tensor)
        pos_img_tensor = transforms.CenterCrop(self.crop_size)(pos_img_tensor)
        neg_img_tensor = transforms.CenterCrop(self.crop_size)(neg_img_tensor)

        # Prepare for positive samples with adversarial attacks
        if self.adv_level == 1:
            pos_img_tensor = adv.apply_instagram_filter(pos_img_tensor)
        elif self.adv_level == 2:
            pos_img_tensor = adv.apply_barrel_distortion(pos_img_tensor)
        elif self.adv_level == 3:
            pos_img_tensor = adv.apply_gaussian_noise(pos_img_tensor)
        elif self.adv_level == 4:
            pos_img_tensor = adv.random_crop_and_resize(pos_img_tensor)
        
        # Make sure all images are in the same device
        anc_img_tensor = anc_img_tensor.to(dev)
        pos_img_tensor = pos_img_tensor.to(dev)
        neg_img_tensor = neg_img_tensor.to(dev)
        
        # Append the label
        anc_label = int(self.camera_images.iloc[idx, 3])
        anc_label_index = self.labels.index(anc_label)
        anc_label_tensor = torch.tensor(anc_label_index, dtype=torch.uint8)
        neg_label = int(self.camera_images.iloc[idx, 5])
        neg_label_index = self.labels.index(neg_label)
        neg_label_tensor = torch.tensor(neg_label_index, dtype=torch.uint8)

        # All tensors are in the same device
        anc_label_tensor = anc_label_tensor.to(dev)
        neg_label_tensor = neg_label_tensor.to(dev)
        
        return {
                "anc_img": anc_img_tensor, 
                "pos_img": pos_img_tensor,
                "neg_img": neg_img_tensor,
                "anc_label": anc_label_tensor,
                "neg_label": neg_label_tensor,
                "metadata": {
                    "anc_img_path": self.camera_images.iloc[idx, 0],
                    "pos_img_path": self.camera_images.iloc[idx, 1],
                    "neg_img_path": self.camera_images.iloc[idx, 2],
                    "anc_device_id": self.camera_images.iloc[idx, 3],
                    "pos_device_id": self.camera_images.iloc[idx, 4],
                    "neg_device_id": self.camera_images.iloc[idx, 5],
                    "anc_FG/BG": self.camera_images.iloc[idx, 6],
                    "neg_FG/BG": self.camera_images.iloc[idx, 7],
                    "train/test": self.camera_images.iloc[idx, 8]
                }
            }
    
    def _load_img(self, img_path):
        img = util.read_img(img_path)
        return util.uint2tensor3(img)