import os

import torch # type: ignore
from torchvision import transforms # type: ignore
from networks import Encoder
from model import mwdcnn
from dataloader.batch_eval_loader import EvaluationDataset, load_img

import prnu
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score # type: ignore
from tqdm import tqdm
from collections import OrderedDict
import pickle
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_PATH = "./saved_models/encoder_final.pth"
DENOISER_PATH = "./saved_models/denoiser_final.pth"
BATCH_SIZE = 32


#######################################################################################################################
#########################################          Helper Functions           #########################################
#######################################################################################################################

# Util functions
def log(*args, **kwargs):
    with open(f'logs/batch_evaluation_{datetime.now().strftime("%m_%d")}.txt', 'a') as f:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs, file=f)

def convert_to_wiener(W):
    W = prnu.rgb2gray(W)
    W = prnu.zero_mean_total(W)
    W_std = W.std(ddof=1)
    W = prnu.wiener_dft(W, W_std).astype(np.float32)
    return W

def get_baseline_score(w1, w2):
    w1 = convert_to_wiener(w1)
    w2 = convert_to_wiener(w2)
    score = prnu.pce(prnu.crosscorr_2d(w1, w2))['pce']
    return score / 5

def get_conv2d_layers(sequential_model):
    conv2d_layers = []
    for layer in sequential_model:
        if isinstance(layer, torch.nn.Conv2d):
            conv2d_layers.append(layer)
    return conv2d_layers

def get_distance(x, y):
    return (x - y).pow(2).sum(1)

#######################################################################################################################
#######################################################################################################################

# Load the pre-trained encoder model
encoder = Encoder(output_size=1024, input_size=(3, 512, 512)).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH), strict=True)
encoder.eval()

# Load the pre-trained denoiser model
denoiser = mwdcnn.make_model()[0]
load_dict = torch.load(DENOISER_PATH)
try:
    denoiser.load_state_dict(load_dict, strict=True)
except RuntimeError:
    new_dict = OrderedDict()
    for key, _ in load_dict.items():    # remove `module.`
        new_dict[key[7:]] = load_dict[key]
    denoiser.load_state_dict(new_dict, strict=True)
denoiser.eval()
denoiser.to(device)

# Load the baseline models
baseline_models = {}

# Load the MWDCNN baseline model
baseline_models["MWDCNN"] = mwdcnn.make_model()[0]
load_dict = torch.load("./saved_models/model_sigma200.pth")
try:
    baseline_models["MWDCNN"].load_state_dict(load_dict, strict=True)
except RuntimeError:
    new_dict = OrderedDict()
    for key, _ in load_dict.items():    # remove `module.`
        new_dict[key[7:]] = load_dict[key]
    baseline_models["MWDCNN"].load_state_dict(new_dict, strict=True)
baseline_models["MWDCNN"].eval()
baseline_models["MWDCNN"].to(device)



#######################################################################################################################
######################################          Main Processing Function           ####################################
#######################################################################################################################

def load_stored_ref():
    global STORED_REF_PATHS, STORED_REF_NOISES, STORED_REF_EMBEDS
    with open('./temps/ref_img_paths.pkl', 'rb') as f:
        STORED_REF_PATHS = pickle.load(f)
    with open('./temps/ref_noises.pkl', 'rb') as f:
        STORED_REF_NOISES = pickle.load(f)
    with open('./temps/ref_embeds.pkl', 'rb') as f:
        STORED_REF_EMBEDS = pickle.load(f)

def get_noise_from_path(img_path):
    # check is the img_path is in the stored paths
    if img_path in STORED_REF_PATHS:
        # get noise from the stored paths
        noise = STORED_REF_NOISES[STORED_REF_PATHS.index(img_path)]
        return noise
    else:
        return None

def get_embed_from_path(img_path):
    # check is the img_path is in the stored paths
    if img_path in STORED_REF_PATHS:
        # get noise from the stored paths
        embed = STORED_REF_EMBEDS[STORED_REF_PATHS.index(img_path)]
        return embed
    else:
        return None

def evaluate(csv_file, adv_combinations=[], batch_size=BATCH_SIZE):
    log(f"Starting evaluation on csv file: {csv_file}")

    level_names = ['Original', 'Color Effects', 'Geometric Trans.', 'Noise Addition', 'Crop + Resize', 'Rotation', 'Perspective', 'iPhone Filters', 'Gaussian Blur', 'Median Blur', 'JPEG Compression']
    if len(adv_combinations) > 0:
        process_names = []
        for i in adv_combinations:
            process_names.append(level_names[i])
        process_names = ', '.join(process_names)
    else:
        process_names = 'Original'

    log(f"Processing Operations Applied: {process_names}")

    # load data
    dataset = EvaluationDataset(csv_file, adv_level=adv_combinations)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize the scores and labels
    scores = []
    labels = []
    baseline_scores = []
    baseline_labels = []

    # iterate through the data
    for i, data in tqdm(enumerate(data_loader)):

        # Reference instance
        ref_img_paths = data["ref_img_paths"] # (10, B)
        ref_imgs = data["ref_imgs"].to(device) # (B, N, C, H, W)
        test_img = data["test_img"].to(device) # (B, C, H, W)
        test_label = data["test_label"].to(device) # (B)

        '''
        test_label == 0: same devices
        test_label == 1: different devices
        '''

        with torch.no_grad():
            # get average embed of ref images and get average noise tensor across N ref images
            ref_embeds = []
            baseline_ref_noises = []
            for batch_index in range(ref_imgs.size(0)):
                # convert list of tuples to numpy array
                ref_img_paths = np.array(ref_img_paths)
                # get the 10 ref_img_paths for this batch
                ref_img_paths_per_batch = ref_img_paths[:, batch_index]
                ref_noise_list = []
                ref_embed_list = []
                any_none = False
                for ref_img_path in ref_img_paths_per_batch:
                    ref_noise = get_noise_from_path(ref_img_path)
                    ref_embed = get_embed_from_path(ref_img_path)
                    if ref_noise is None or ref_embed is None:
                        any_none = True
                        break
                    # convert to tensor
                    ref_noise = torch.from_numpy(ref_noise)
                    ref_embed = torch.from_numpy(ref_embed)
                    ref_noise_list.append(ref_noise)
                    ref_embed_list.append(ref_embed)
                # if any of the ref_noise or ref_embed is None, we need to compute the noise and embed
                if not any_none:
                    ref_noise_tensor = torch.stack(ref_noise_list).to(device)
                    ref_embed_tensor = torch.stack(ref_embed_list).to(device)
                    baseline_ref_noises.append(ref_noise_tensor.mean(0))
                    ref_embeds.append(ref_embed_tensor.mean(0))
                else:
                    ref_img = ref_imgs[batch_index] # (N, C, H, W)
                    # for baseline
                    baseline_ref_denoised = baseline_models["MWDCNN"](ref_img)
                    baseline_ref_noise = ref_img - baseline_ref_denoised
                    baseline_ref_noises.append(baseline_ref_noise.mean(0)) # append the average noise tensor
                    # for our method
                    ref_denoised = denoiser(ref_img)
                    ref_noise = ref_img - ref_denoised            
                    ref_embed = encoder(0.5 + ref_noise)
                    ref_embeds.append(ref_embed.mean(0)) # append the average embedding
            baseline_ref_noises = torch.stack(baseline_ref_noises)
            ref_embeds = torch.stack(ref_embeds)

            # get the baseline noise tensor for test image
            baseline_test_denoised = baseline_models["MWDCNN"](test_img)
            baseline_test_noise = test_img - baseline_test_denoised

            # get embed of test image
            test_denoised = denoiser(test_img)
            test_noise = test_img - test_denoised
            test_embed = encoder(0.5 + test_noise)

        # calculate pairwise distance         
        scores += list(get_distance(ref_embeds, test_embed).cpu().numpy())
        labels += list(test_label.cpu().numpy())

        # get the average of the noise tensor for baseline
        for batch_index in range(ref_imgs.size(0)):
            baseline_ref_noise_tensor = baseline_ref_noises[batch_index]
            baseline_ref_noise_tensor = baseline_ref_noise_tensor.cpu().numpy().transpose(1, 2, 0)            
            baseline_test_noise_tensor = baseline_test_noise[batch_index].cpu().numpy().transpose(1, 2, 0)
            baseline_scores.append(get_baseline_score(baseline_ref_noise_tensor, baseline_test_noise_tensor))
            baseline_labels.append(1. - test_label[batch_index].cpu().numpy()) # we need to reverse the label for baseline
    
    baseline_y_true, baseline_y_pred = (baseline_labels, baseline_scores)
    y_true, y_pred = (labels, scores)
    
    # store the baseline_y_true and baseline_y_pred for future use using pickle
    out_root_dir = '../../Evaluation_Results/'
    out_dir_name = csv_file.split('/')[-1].split('.')[0]
    adv_name = '_'.join([str(i) for i in adv_combinations])
    out_dir_name = f"{out_dir_name}_adv_{adv_name}"
    out_dir = os.path.join(out_root_dir, out_dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, f"baseline_y_true.pkl"), "wb") as f:
        pickle.dump(baseline_y_true, f)
    with open(os.path.join(out_dir, f"baseline_y_pred.pkl"), "wb") as f:
        pickle.dump(baseline_y_pred, f)
    with open(os.path.join(out_dir, f"y_true.pkl"), "wb") as f:
        pickle.dump(y_true, f)
    with open(os.path.join(out_dir, f"y_pred.pkl"), "wb") as f:
        pickle.dump(y_pred, f)

    log(f"AUC (ours): {roc_auc_score(y_true, y_pred):.2f}, AUC (baseline): {roc_auc_score(baseline_y_true, baseline_y_pred):.2f}")


def __process_ref_imgs(csv_path):
    df = pd.read_csv(csv_path, sep="\t")
    ref_img_paths = list(df["ref_img_paths"].apply(lambda x: x.split(",")).explode().unique())
    print(f"Total number of reference images: {len(ref_img_paths)}")

    ref_embeds = []
    ref_noises = []
    for ref_img_path in tqdm(ref_img_paths):
        with torch.no_grad():
            ref_img_tensor = load_img(ref_img_path)
            ref_img_tensor = transforms.CenterCrop(512)(ref_img_tensor) # Center crop the image
            ref_img_tensor = ref_img_tensor.unsqueeze(0)
            ref_img_tensor = ref_img_tensor.to(device)
            # for baseline
            baseline_ref_denoised = baseline_models["MWDCNN"](ref_img_tensor)
            baseline_ref_noise = (ref_img_tensor - baseline_ref_denoised).squeeze(0)
            ref_noises.append(baseline_ref_noise.cpu().numpy())
            # for our method
            ref_denoised = denoiser(ref_img_tensor)
            ref_noise = ref_img_tensor - ref_denoised
            ref_embed = encoder(0.5 + ref_noise)
            ref_embeds.append(ref_embed.squeeze(0).cpu().numpy())
    
    # save the ref_img_paths and ref_img_noises using pickle
    out_root_dir = './temps/'
    with open(os.path.join(out_root_dir, 'ref_img_paths.pkl'), 'wb') as f:
        pickle.dump(ref_img_paths, f)
    with open(os.path.join(out_root_dir, 'ref_noises.pkl'), 'wb') as f:
        pickle.dump(ref_noises, f)
    with open(os.path.join(out_root_dir, 'ref_embeds.pkl'), 'wb') as f:
        pickle.dump(ref_embeds, f)


#######################################################################################################################
##############################################            Main               ##########################################
#######################################################################################################################

if __name__ == "__main__":
    # Load CSV file
    CSV_FILE = f'../../Evaluation/overall.csv'
    # Pre-compute the reference images
    log(f"Processing reference images for csv file: {CSV_FILE}")
    __process_ref_imgs(CSV_FILE)

    # Load the stored reference images
    load_stored_ref()

    # Evaluation Configuration [Please feel free to change the configurations]
    BATCH_SIZE = 32 # please modify based on your GPU memory
    list_of_adv_combinations = [
        [0], # Original
        [1], # Color Effects
        [2], # Geometric Trans.
        [3], # Noise Addition
        [4], # Crop + Resize
        [5], # Rotation
        [6], # Perspective
        [7], # iPhone Filters
        [8], # Gaussian Blur
        [9], # Median Blur
        [10], # JPEG Compression
        [1, 2, 3, 4], # Example combinations
    ] # Please feel free to add more combinations or remove some of them

    # Start evaluation
    log(f"Starting evaluation on csv file: {CSV_FILE}")
    for evaluat_combination in list_of_adv_combinations:
        evaluate(CSV_FILE, adv_combinations=evaluat_combination, batch_size=BATCH_SIZE)
    