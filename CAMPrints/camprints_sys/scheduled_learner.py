import torch # type: ignore
import torch.nn.functional as F # type: ignore
from torch.optim import lr_scheduler # type: ignore
from torch.utils.data import ConcatDataset, Subset, DataLoader # type: ignore
from sklearn import metrics # type: ignore
import numpy as np
import model.mwdcnn as mwdcnn
from networks import Encoder
from datetime import datetime
from dataloader.triplet_loader import CameraDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Util functions
def log(*args, **kwargs):
    with open(f'logs/train_scheduled_{datetime.now().strftime("%m_%d")}.txt', 'a') as f:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs, file=f)

def triplet_loss(anchor, positive, negative, margin=1):
    # Compute the Euclidean distances between the anchor and positive embeddings
    pos_dist = F.pairwise_distance(anchor, positive)
    # Compute the Euclidean distances between the anchor and negative embeddings
    neg_dist = F.pairwise_distance(anchor, negative)
    # Compute the triplet loss
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def get_distance(x, y):
    return (x - y).pow(2).sum(1)

def load_datasets():
    # Load dataset
    dataset_dict = {}
    for i in range(5):
        dataset_dict[f"Train_{i}"] = CameraDataset(train=True, adv_level=i, crop_size=512)
        dataset_dict[f"Val_{i}"] = CameraDataset(train=False, adv_level=i, crop_size=512)
    log('Load datasets')
    return dataset_dict

def concat_train_datasets(dataset_dict, focus_level, levels):
    datasets = []
    for level in levels:
        if level == focus_level:
            # use the entire dataset
            datasets.append(dataset_dict[f"Train_{level}"])
        else:
            # use 30% of the dataset
            dataset = dataset_dict[f"Train_{level}"]
            subset_indices = np.random.choice(len(dataset), size=int(0.3 * len(dataset)), replace=False)
            # Create a subset dataset
            subset = Subset(dataset, subset_indices)
            datasets.append(subset)
    return ConcatDataset(datasets)

def get_val_loader(dataset_dict, levels, batch_size):
    val_loader_list = []
    for level in levels:
        val_dataset = dataset_dict[f"Val_{level}"]
        # get subset
        subset_indices = np.random.choice(len(val_dataset), size=int(0.2 * len(val_dataset)), replace=False)
        subset = Subset(val_dataset, subset_indices)
        val_loader_list.append(DataLoader(subset, batch_size=batch_size, shuffle=False))
    return val_loader_list

def evaluate(denoiser, model, loader, remark=''):
    # Validation
    model.eval()
    denoiser.eval()

    with torch.no_grad():
        val_loss = 0
        pos_dists = []
        neg_dists = []
        for data in loader:
            # Load data to device
            anchors = data['anc_img'].to(device)
            positives = data['pos_img'].to(device)
            negatives = data['neg_img'].to(device)

            # Get the noise_residuals from the denoiser
            anc_denoised = denoiser(anchors)
            anc_noise = (anchors - anc_denoised)
            anc_noise = 0.5 + anc_noise
            pos_denoised = denoiser(positives)
            pos_noise = (positives - pos_denoised)
            pos_noise = 0.5 + pos_noise
            neg_denoised = denoiser(negatives)
            neg_noise = (negatives - neg_denoised)
            neg_noise = 0.5 + neg_noise

            # Forward pass
            anc_embeds = model(anc_noise)
            pos_embeds = model(pos_noise)
            neg_embeds = model(neg_noise)

            # Compute the loss
            loss = triplet_loss(anc_embeds, pos_embeds, neg_embeds)

            # Accumulate the loss
            val_loss += loss.item()

            # Append the embeddings to the list
            pos_dists += list(get_distance(anc_embeds, pos_embeds).cpu().numpy())
            neg_dists += list(get_distance(anc_embeds, neg_embeds).cpu().numpy())

        # Calculate the ROC-AUC score
        pos_dists = torch.tensor(pos_dists)
        neg_dists = torch.tensor(neg_dists)
        all_dists = torch.cat([pos_dists, neg_dists], dim=0)
        labels = torch.cat([torch.zeros_like(pos_dists), torch.ones_like(neg_dists)], dim=0)
        roc_auc = metrics.roc_auc_score(labels, all_dists)
        
        # Log the loss
        log(f'Epoch [{epoch}/{num_epochs}], Validation {remark} Loss: {val_loss/len(loader)}, AUROC: {roc_auc}')

        return val_loss/len(loader), roc_auc

def train_epoch(epoch, denoiser, model, train_loader, optimizer):
    model.train()
    denoiser.train()
    train_loss = 0
    for i, data in enumerate(train_loader):
        # Load data to device
        anchors = data['anc_img'].to(device)
        positives = data['pos_img'].to(device)
        negatives = data['neg_img'].to(device)

        # Get the noise_residuals from the denoiser
        anc_denoised = denoiser(anchors)
        anc_noise = (anchors - anc_denoised)
        anc_noise = 0.5 + anc_noise
        pos_denoised = denoiser(positives)
        pos_noise = (positives - pos_denoised)
        pos_noise = 0.5 + pos_noise
        neg_denoised = denoiser(negatives)
        neg_noise = (negatives - neg_denoised)
        neg_noise = 0.5 + neg_noise

        # Forward pass
        anc_embeds = model(anc_noise)
        pos_embeds = model(pos_noise)
        neg_embeds = model(neg_noise)

        # Compute the loss
        loss = triplet_loss(anc_embeds, pos_embeds, neg_embeds)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        train_loss += loss.item()

        # Log the loss
        if (i+1) % 10 == 0:
            log(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
    
    return train_loss / len(train_loader)


# Load the pre-trained MWDCNN model
denoiser = mwdcnn.make_model()[0]
load_dict = torch.load('saved_models/denoiser_final.pth')
try:
    denoiser.load_state_dict(load_dict, strict=True)
except RuntimeError:
    from collections import OrderedDict
    new_dict = OrderedDict()

    for key, _ in load_dict.items():    # remove `module.`
        new_dict[key[7:]] = load_dict[key]

    denoiser.load_state_dict(new_dict, strict=True)
denoiser.to(device)
# freeze most layers of denoiser, only keep the last two CNN blocks trainable
for param in denoiser.parameters():
    param.requires_grad = False  
for param in denoiser.conv_block2.parameters():
    param.requires_grad = True
for param in denoiser.conv2.parameters():
    param.requires_grad = True
log('Load pre-trained denoiser [MWDCNN] to GPU')

# Training parameters
learning_rate = 1e-3
start_epoch = 10
num_epochs = 100
batch_size = 32
latent_dim = 1024

# Initialize and load model using default parameters
model = Encoder(output_size=latent_dim, input_size=(3, 512, 512)).to(device)
model.load_state_dict(torch.load('saved_models/encoder_final.pth'))
params = list(denoiser.parameters()) + list(model.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
log('Initialize and load base networks to GPU')

# Initialize the epoch scheduler
epoch_data_scheduler = 0

# Training loop
log('Start training')

for epoch in range(0, start_epoch):
    scheduler.step()

for epoch in range(start_epoch, num_epochs):
    scheduler.step()

    # Load datasets based on the epoch data scheduler
    dataset_dict = load_datasets()
    train_dataset = concat_train_datasets(dataset_dict, epoch_data_scheduler, [0, 1, 2, 3, 4])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader_list = get_val_loader(dataset_dict, [0, 1, 2, 3, 4], batch_size=batch_size)
    log(f'Epoch [{epoch}/{num_epochs}], Adv Level Focused: Level {epoch_data_scheduler}')

    # Train the model
    train_loss = train_epoch(epoch, denoiser, model, train_loader, optimizer)
    log(f'Training Loss: {train_loss}')

    # Evaluate the model
    val_losses = []
    roc_aucs = []
    for level, val_loader in enumerate(val_loader_list):
        val_loss, roc_auc = evaluate(denoiser, model, val_loader, remark=f'Level {level}')
        val_losses.append(val_loss)
        roc_aucs.append(roc_auc)
    
    log_msg = f'Training Loss: {train_loss}\n'
    for level, (val_loss, roc_auc) in enumerate(zip(val_losses, roc_aucs)):
        log_msg += f'Val Loss Level {level}: {val_loss}, AUROC Level {level}: {roc_auc}\n'
    
    # Save the model for each epoch
    log(f'Saving denoiser to checkpoints/denoiser_ep{epoch:02d}.pth')
    torch.save(denoiser.state_dict(), f'checkpoints/denoiser_ep{epoch:02d}.pth')
    log(f'Saving encoder to checkpoints/encoder_ep{epoch:02d}.pth')
    torch.save(model.state_dict(), f'checkpoints/encoder_ep{epoch:02d}.pth')
    
    # Increment the epoch scheduler
    epoch_data_scheduler += 1
    if epoch_data_scheduler > 4:
        # reset the scheduler to 0 if beyond 0 to 4
        epoch_data_scheduler = 0

    

