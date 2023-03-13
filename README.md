# WLSegNet

This is the official code for "A Language-Guided Benchmark for Weakly Supervised Open Vocabulary Semantic Segmentation".

<p align="center">
  <img src="assets/setting.png" width="80%"/><br>
</p>

* First method to explore multiple and related Open Vocabulary Semantic Segmentation inductive tasks in a weakly supervised setting without using external datasets and fine-tuning
* First method to handle weakly supervised generalized zero-shot segmentation, zero-shot segmentation and few-shot segmentation with a single training procedure using a frozen vision-language model 
* Propose a novel and scalable mean instance aware prompt learning that generates highly generalizable prompts, handles domain shift across the datasets and generalizes efficiently to unseen classes
* The flexible design allows easy modification and optimization of different components as and when required  
* The proposed method beats existing weakly supervised baselines by large margins while being competitive with pixel-based methods


## Installation and setup

### Installing dependencies

To install required packages for running this code, follow the instructions below

```
git clone https://github.com/mustafa1728/WLSegNet.git
cd WLSegNet
conda create --name WLSegNet # (optional, for making a conda environment)
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
cd third_party/CLIP
python -m pip install -Ue .
```

Please chose the pytorch and cudatoolkit versions according to your CUDA environment. Refer pytorch installation  instructions [here](https://pytorch.org/get-started/locally/). For more details on installing detectron2 and selecting necessary options based on your CUDA environment, please refer detectron2 installation instructions [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).


### Data Preparation

We experiment with PASCAL VOC 2012 [[page](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)] and MS COCO 2014 [[page](https://cocodataset.org/#home)] datasets. Download the images and corresponding annotations from [here](https://drive.google.com/file/d/1uh5bWXvLOpE-WZUUtO77uwCB4Qnh6d7X/view) and [here](https://cocodataset.org/#download) for pascal and coco respectively. The code expects the following directory structure:

```
- MIAPNet
   - datasets
      - VOC2012
         - JPEGImages
      - coco
         - JPEGImages
```

You may need to use a different directory structure, in which case, change necessary paths in [`register_pascal`](./mask_former/data/datasets/register_voc_seg.py) and [`register_coco`](./mask_former/data/datasets/register_coco.py).

Download our generated pixel pseudo labels  for different fold of both PASCAL and COCO from [here](https://drive.google.com/drive/folders/1gI4XSlYhmGHSv6YlWLUcOmYntWiMAHxr?usp=sharing). Alternatively, you may generate the pseudo labels youself by following the training and psuedo label generation procedure outlined in [L2G](https://github.com/PengtaoJiang/L2G).

After downloading the required datasets, run [`preprocess`](./preprocess/) files to generate validation labels for different splits in ZSS and the 1-way and 2-way settings in FSS. 

## Training 

To train your models, make necessary changes in the configs of your choice. The configs of ZSS and FSS are in [`voc`](./configs/voc/) for PASCAL VOC and in [`coco`](./configs/coco/) for COCO. Additional ablation experiments can be run using config files in [`ablations`](./configs/ablations/).

A generic command to run a paritcular experiment corresponding to a config is:

```shell
python3 train_net.py --config-file <config-path>
```

For example, to run WZSS on fold 3 of PASCAL VOC, run the following command:
```shell
python3 train_net.py --config-file ./configs/voc/wzss/fold0.yaml
```

You can provide command line arguments to modify certain config entries like this:
```shell
python3 train_net.py --config-file <config-path> --num-gpus 6 OUTPUT_DIR <output-path> MODEL.WEIGHTS <weight-init-path>
```

### ZSS training

To train a model on a particular setting, you need to first run the learning prompts experiment and then run the zss experiment using the context vectors and weights learnt in the first part. For example, for PASCAL VOC fold 0, the commands are: 

1. Run Prompt Learning
    ```shell
    python3 train_net.py --config-file ./configs/voc/prompt_learn/fold0.yaml OUTPUT_DIR <output-path> 
    ```
2. Add the path to the learnt weights in WZSS config [here](./configs/voc/wzss/fold0.yaml). Change the following entries accordingly:
    ```yaml
    MODEL:
        CLIP_ADAPTER: 
            PROMPT_CHECKPOINT: <output-path>/model_final.pth
            REGION_CLIP_ADAPTER:
                PROMPT_CHECKPOINT: <output-path>/model_final.pth
    ```
3. Run the WZSS experiment
    ```shell
    python3 train_net.py --config-file ./configs/voc/wzss/fold0.yaml
    ```

Alternatively, you may use pretrained model weights released [here](https://drive.google.com/drive/folders/1A0S7gr3zwHD_LqYqHZ-dIvcoKx4uO0Al?usp=sharing).

## Evaluation 

By default, a model being trained for wzss  gets evaluated at periodic intervals. To evaluate a trained model for wzss separately, simply run:

```shell
python3 train_net.py --config-file <config-path> --eval-only --resume
```

You may set the path to a trained model weight in command line like this:
```shell
python3 train_net.py --config-file <config-path> --eval-only MODEL.WEIGHTS <weight-path>
```

### WZSS 5i evaluation

The command to evaluate WZSS in PASCAL 5i fold 0 is:

```shell
python3 train_net.py --config-file ./configs/voc/wzss_5i/fold0.yaml --eval-only MODEL.WEIGHTS <weight-path>
```

where `<weight-path>` points to the path of the save model during WZSS training of fold 0 PASCAL VOC. Some care needs to be taken to make sure that folds of trained models match those of testing datasets.

### WZSS cross dataset evaluation

The command to evaluate COCO to PASCAL WZSS fold 0 is:

```shell
python3 train_net.py --config-file ./configs/voc/wzss_cross/fold0.yaml --eval-only MODEL.WEIGHTS <weight-path>
```

where `<weight-path>` points to the path of the save model during WZSS training of fold 0 PASCAL VOC. Some care needs to be taken to make sure that folds of trained models match those of testing datasets.


### WFSS evaluation

To evaluate WFSS, a model trained in the WZSS setting can be used directly. The command to run WFSS on PASCAL VOC fold 0 is:

```shell
python3 train_net.py --config-file ./configs/voc/wfss/fold0.yaml --eval-only MODEL.WEIGHTS <weight-path>
```

where `<weight-path>` points to the path of the save model during WZSS training of fold 0 PASCAL VOC. Some care needs to be taken  to make sure that fold 0 WZSS models are testing for fold 0 WFSS only. 

Similarly, experiments for the 2-way setting can be run with the same trained model:
```shell
python3 train_net.py --config-file ./configs/voc/wfss_2way/fold0.yaml --eval-only MODEL.WEIGHTS <weight-path>
```

## Visualisation

###  WZSS

During evaluation, set the `DATASETS.VIS_MULTIPLIER` entry to 1 in the corresponding config. Segmentation maps for seen and unseen categories will be generated in `OUTPUT_DIR/evaluation/pred_vis` directory. Run [`visualize.py`](./visualize.py) after changing the `dir_name` and `vis_dir_name` accordingly.

### WFSS

During evaluation, set the `DATASETS.VIS_MULTIPLIER` entry to 255 for 1-way and 127 for 2-way in the corresponding configs. Segmentation maps for unseen categories will be generated in `OUTPUT_DIR/evaluation/pred_vis` directory. For 1-way, these will be binary maps and for 2-way these will have 0, 127 and 254 as the three distinct values.

## Acknowledgement

We thank the authors of [`Maskformer`](https://github.com/facebookresearch/MaskFormer), [`CLIP`](https://github.com/openai/CLIP) and [`Simple Baseline`](https://github.com/MendelXu/zsseg.baseline) for their awesome works. This repo benefits greatly from them.
