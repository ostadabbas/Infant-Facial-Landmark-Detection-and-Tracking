# InfAnFace: Bridging the Infant--Adult Domain Gap in Facial Landmark Estimation in the Wild


## Introduction 
This is the official repository for:

Wan, M., Zhu, S., Luan, L., Prateek, G., Huang, X., Schwartz-Mette, R., Hayes, M., Zimmerman, E., & Ostadabbas, S.  "InfAnFace: Bridging the infant-adult domain gap in facial landmark estimation in the wild." *26th International Conference on Pattern Recognition* (ICPR 2022). [[arXiv link](https://arxiv.org/abs/2110.08935)]

In this paper, we introduce:

* **InfAnFace**, the first **Inf**ant **An**notated **Face**s dataset, consisting of 410 images of infant faces with labels for 68 facial landmark locations and various pose attributes. Here are some sample images and ground truth labels, sorted by subset or attributes described in our paper:

<div align="center">
<img src="images/landmark-samples.png" alt="InfAnFace landmark samples" width="400"/>
</div>

* State-of-the-art facial landmark estimation models designed specifically for infant faces, based on the [HRNet](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) architecture. Here are sample predictions from our HRNet-R90JT model (bottom) compared to two other models, [3FabRec](https://github.com/browatbn2/3FabRec) and [HRNet](https://github.com/HRNet/HRNet-Facial-Landmark-Detection):

<div align="center">
<img src="images/prediction-samples.png" alt="Infant landmark predictions" width="800"/>     
</div>

See below for instructions for accessing InfAnFace and training and testing with our models.


## Licence

By downloading or using any of the datasets provided by the ACLab, you are agreeing to the “Non-commercial Purposes” condition. “Non-commercial Purposes” means research, teaching, scientific publication and personal experimentation. Non-commercial Purposes include use of the Dataset to perform benchmarking for purposes of academic or applied research publication. Non-commercial Purposes does not include purposes primarily intended for or directed towards commercial advantage or monetary compensation, or purposes intended for or directed towards litigation, licensing, or enforcement, even in part. These datasets are provided as-is, are experimental in nature, and not intended for use by, with, or for the diagnosis of human subjects for incorporation into a product.

Users of this repository must abide by the respective licenses of any code included from other sources.


## The InfAnFace dataset

**InfAnFace**, the **Inf**ant **An**notated **Face**s dataset, is the first public dataset of infant faces with facial landmark annotations. It consists of 410 images of infant faces with labels for 68 facial landmark coordinates and various pose attributes. The dataset can be accessed [at this site](https://coe.northeastern.edu/Research/AClab/InfAnFace/) or downloaded as a [ZIP file](https://coe.northeastern.edu/Research/AClab/InfAnFace.zip). The file structure is as follows:

````
infanface
-- readme.md
-- images
   |-- ads
   |-- google
   |-- google2
   |-- youtube
   |-- youtube2
-- labels.csv
````

The images, in `/images`, are organized into five batches, which were collected and annotated by our team separately: 

* `/ads` (104 images) contains infant formula advertisement video stills sourced from YouTube,
* `/google` (100 images) and `/google2` (51 images) contain photos sourced from Google Images, and
* `/youtube` (100 images) and `/youtube2` (55 images) contain video stills sourced from YouTube.

Annotated labels can be found in `labels.csv`. These include:

* **68 facial landmark annotations** following the Multi-PIE layout (like [300-W](https://ibug.doc.ic.ac.uk/resources/300-W/)),
* binary **attribute labels** (*turned*, *occluded*, *expressive*, and *tilted*, as defined in our paper), and
* a division of the images into Train and Test sets, and a division of the Test set into Common and Challenging sets, as described below.


#### Train, Test, Common, and Challenging subsets of InfAnFace

In order to facilitate training experiments and interpretable testing, we divide InfAnFace into a number of subsets, loosely inspired by the eponymous 300-W subsets. 

InfAnFace (410 images) is split into suggested training and tests sets based on our collection batches to ensure some level of independence: 

* **InfAnFace Train** (210 images) includes the `ads`, `google2`, and `youtube2`, while
* **InfAnFace Test** (200 images) includes the `google` and `youtube`.

InfAnFace Test (200 images) is further split into:

* **InfAnFace Common** (80 images), an easier test set, consisting of subjects for which all four of our binary attributes are false, i.e., images with infants which are not turned, not occluded, not expressive, *and* not tilted; and
* **InfAnFace Challenging** (120 images), a difficult test set, consisting of subjects for which at least one of our four binary attributes holds true.


## Facial landmark estimation models

Here are instructions for training the state-of-the-art infant facial landmark estimation models described in our paper, and for making landmark predictions using these or previously pre-trained models. We include all of the HRNet-based models from our paper. The best all-around model is **HRNet-R90JT**.

Our code was developed and tested with Python 3.6, PyTorch 1.0.0, with NVIDIA P100 GPUs under CUDA 9.0. Many of these instructions overlap with the installation and use of HRNet, so their [guide](https://github.com/HRNet/HRNet-Facial-Landmark-Detection/blob/master/README.md) can serve as a fall-back to ours.

#### Basic installation

1. [Install PyTorch 1.0](https://pytorch.org/)
2. Clone the project `git clone MISSING LINK`
3. Install requirements `pip install -r requirements.txt`

*Those in just using our pretrained models can skip the next two sections.*

#### Model training 

4. Add the checkpoint for the HRNet model trained for facial landmark estimation on 300-W images, [HR18-300W.pth](https://onedrive.live.com/?authkey=%21AMkPimlmClRvmpw&cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21112&parId=F7FD0B7F26543CEB%21105&o=OneUp), to `/hrnetv2_pretrained`.
5. Install 300-W and/or InfAnFace image data by placing subfolders from those datasets (e.g., `/afw`, `/frgc`, etc. from 300-W, or `/ads`, `/google`, etc. from InfAnFace) inside `/data/images`. The CSV files already included in subdirectories of `/data` contain facial landmark annotations for 300-W and InfAnFace, and also define subsets of interest of both datasets, e.g., 300-W Test or InfAnFace Challenging. (Other datasets can be installed by mimicking these configurations.) The final data directory structure should look something like this:

````
Infant-Facial-Landmark-Detection-and-Tracking
-- lib
-- experiments
-- tools
-- data
   |-- 300w
   |   |-- 300w_test.csv
   |   |-- 300w_train.csv
   |   |-- 300w_valid.csv
   |   |-- 300w_valid_challenge.csv
   |   |-- 300w_valid_common.csv
   |-- infanface
   |   |-- ads.csv
   |   |-- google.csv
   |   |-- google2.csv
   |   |-- youtube.csv
   |   |-- youtube2.csv
   |-- images
   |   |-- ads
   |   |-- afw
   |   |-- frgc
   |   |-- google
   |   |-- etc.
````

6. Train the base HRNet model using 300-W only data: `python -u tools/train.py --cfg experiments/300w/hrnet.yaml`
7. Train our fine-tuned HRNet-R90 model with InfAnFace data on top of a HRNet training checkpoint from Step 6:
`python -u tools/finetune.py --cfg experiments/300w/hrnet-ft.yaml --model-file output/300W/hrnet/checkpoint_40.pth`
8. Train our best HRNet-R90JT model using joint 300-W and InfAnFace data: `python -u tools/train.py --cfg experiments/300w/hrnet-r90jt.yaml`

Output checkpoints are produced in `/output`. Each HRNet-based model in our paper corresponds to a YAML configuration file in `/experiments`. Other models can be trained by modifying these configuration files.

#### Landmark estimation with trained models

9. Test a checkpoint of the base HRNet model on 300-W Test data: `python -u tools/test.py --cfg experiments/300w/hrnet.yaml --model-file output/300W/hrnet/checkpoint_40.pth`
10. Test a checkpoint of our best HRNet-R90JT model on InfAnFace Test data: `python -u tools/test.py --cfg experiments/300w/hrnet-r90jt.yaml --model-file output/300W/hrnet-r90jt/checkpoint_40.pth`

Landmark predictions in text format are saved in the corresponding model directory in `/output`. The reported NME results use the interocular normalization. Note that the test set is specified by the YAML configuration file in `/experiments`. 

#### Pretrained model installation and landmark estimation

11. Add the checkpoints for our modified HRNet models, which can be downloaded [here](https://drive.google.com/drive/folders/1Gj0aec2MmsRRLpa5JGqxi4sA1G4Sr79Z?usp=sharing), to `/infanface_pretrained`
12. Test the pretrained checkpoint of our best HRNet-R90JT model on InfAnFace Test data: `python -u tools/test.py --cfg experiments/300w/hrnet-r90jt.yaml --model-file infanface_pretrained/hrnet-r90jt.pth`

As above: Landmark predictions in text format are saved in the corresponding model directory in `/output`. The reported NME results use the interocular normalization. Note that the test set is specified by the YAML configuration file in `/experiments`. 

 
## Citation
Here is a BibTeX entry for our ICPR 2022 paper:
````
@inproceedings{WanICPR2022,
  title={{InfAnFace}: {Bridging} the infant-adult domain gap in facial landmark estimation in the wild},
  author={Michael Wan and Shaotong Zhu and Lingfei Luan and Prateek Gulati and Xiaofei Huang and Rebecca Schwartz-Mette and Marie Hayes and Emily Zimmerman and Sarah Ostadabbas},
  booktitle = {2022 {International} {Conference} on {Pattern} {Recognition} ({ICPR})},
  year={2022}
}
````

