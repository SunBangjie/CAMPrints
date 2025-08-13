# CAMPrints: Leveraging the "Fingerprints" of Digital Cameras to Combat Image Theft
This repository accompanies the paper published at [ACM MobiSys 2025](https://www.sigmobile.org/mobisys/2025/). 
Please access the full paper here: [CAMPrints: Leveraging the "Fingerprints" of Digital Cameras to Combat Image Theft](https://sunbangjie.github.io/files/CAMPrints.pdf).

## Abstract
Photo sharing is increasingly popular, driven by social media platforms like Instagram and services such as Flickr and Google Photos. However, this growth has been accompanied by significant issues, particularly image theft. To address this issue, we introduce CAMPrints, a robust system for detecting image theft. CAMPrints verifies whether edited images found online contain camera fingerprints matching those of user-provided reference images. The system overcomes the challenges of identifying images altered by diverse image processing operations. We select a small yet representative set of operations by categorizing them based on their impact on pixel values and locations. A deep-learning model is trained to recognize and compare camera noise patterns pre- and post-editing. We conduct real-world evaluations involving 36 cameras across eight make-and-model combinations, along with over 40 image processing operations applied to more than 4,000 images. CAMPrintsachieves an average AUC of 0.92, significantly outperforming the state-of-the-art methods by up to 1.8 times.

## Environment Set Up
Please use the following commands to set up the Python virtual environment.
```Python
>> cd CAMPrints
>> conda create -n camprints python=3.9
>> conda activate camprints
>> pip install torch==2.2.2+cu121
torchvision==0.17.2+cu121
-f https://download.pytorch.org/whl/
torch_stable.html
>> pip install -r requirements.txt
```

## Dataset
You can access the dataset [here](https://drive.google.com/file/d/12wZxxVIH8ZTVuun7IGUNzn6Gs2ptrrOq/view?usp=sharing). Please note that this dataset is a subset of the [SOCRatES](https://socrates.eurecom.fr/) dataset. If you wish to use the dataset, please visit the [SOCRatES](https://socrates.eurecom.fr/) website to obtain in the license agreement.

Once you download the dataset `Dataset.zip`, please unzip and put the folder `Dataset\` in the root directory.

## Pre-trained Model
You can access the pre-trained encoder model [here](https://drive.google.com/file/d/1o5huB8RoKGLe2jHWOD7xVAadBDy14F7q/view?usp=sharing). Please place the `encoder_final.pth` file in the directory `CAMPrints/camprints_sys/saved_models/`, together with two other pre-trained models.

## Evaluation
We provide a script to run the experiments automatically. Use the following commands:
```Python
>> cd CAMPrints/camprints_sys/
>> python batch_evaluation.py
```
The results will be saved in the folder `Evaluation_Results/`. Different combinations of image processing operations can be applied to the images, with each combination taking approximately 20 minutes. The script `batch_evaluation.py` includes a subset of these combinations. This configuration is defined through the following variable:
```Python
line 300: list_of_adv_combinations = [ ... ]
```
The complete list of combinations is provided in the following Table. The full evaluation, covering all combinations, is expected to take approximately 10 hours.

| Type of Combs. | Combs. of Ops. | No. of Combs. |
|----------|----------|----------|
| Single Op.    | [0], [1], [2], [3], [4],[5], [6], [7], [8], [9], [10]   | 11   |
| Two Ops.    |  [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]   | 6   | 
| Three Ops. | [1, 2, 3], [1, 2, 4], [1, 3, 2], [1, 3, 4], [2, 1, 3], [2, 3, 1], [2, 3, 4], [3, 1, 2], [3, 2, 1] | 9
| Four Ops.  | [1, 2, 3, 4], [1, 3, 2, 4], [3, 1, 4, 2], [4, 2, 3, 1], [4, 3, 2, 1], [7, 6, 8, 5] | 6

## Expected Results
We provide a Jupyter notebook, `Plot_Results.ipynb` for aggregating and visualizing the evaluation results. 
### Visualizing Expected Results. 
Running all cells in the notebook without any modifications will generate the expected results, displaying ROC curves similar to those in Figure 9 in the full paper.
### Visualizing Evaluation Results.
After Running Experiments. To visualize evaluation results after running the experiments, update the following code in the Jupyter notebook. 
In the third cell:
```Python
if method == "CAMPrints" or method == "MWDCNN":
# root_folder = "./Expected_Results/..."
root_folder = "./Evaluation_Results"In the last cell:# for method in ["CAMPrints", "DRUNET", ...]:
for method in ["CAMPrints", "MWDCNN"]:
```
The code above reflects the necessary modifications. After making these changes, run all cells again to generate the updated plot, which will include results for both CAMPrints and the baseline model (MWDCNN). Note that we do not provide code for the two additional baselines, DRUNET and the Wiener method.

## Citation
If **_CAMPrints_** contributes to your work, we would appreciate a citation to our paper.

### BibTeX
```bibtex
@inproceedings{sun2025camprints,
  author    = {Bangjie Sun and Mun Choon Chan and Jun Han},
  title     = {CAMPrints: Leveraging the ``Fingerprints'' of Digital Cameras to Combat Image Theft},
  booktitle = {Proceedings of the 23rd Annual International Conference on Mobile Systems, Applications and Services (MobiSys~'25)},
  year      = {2025},
  month     = jun,
  address   = {Anaheim, CA, USA},
  publisher = {ACM},
  location  = {Anaheim, CA, USA},
  pages     = {1--14},
  doi       = {10.1145/3711875.3729158},
  url       = {https://doi.org/10.1145/3711875.3729158},
  note      = {14 pages}
}
```

### ACM Reference Format
```
Bangjie Sun, Mun Choon Chan, and Jun Han. 2025. CAMPrints: Leveraging the “Fingerprints” of Digital Cameras to Combat Image Theft. In The 23rd Annual International Conference on Mobile Systems, Applications and Services (MobiSys ’25), June 23–27, 2025, Anaheim, CA, USA. ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3711875.3729158
```
