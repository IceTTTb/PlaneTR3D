# PlaneTR: Structure-Guided Transformers for 3D Plane Recovery

This is the official implementation of our [ICCV 2021 paper](https://arxiv.org/abs/2107.13108)


<img src="misc/network.jpg" width="600">


## News
- 2023.2.9: Upload the depth evaluation code on the NYUV2 dataset. 
- 2021.08.08: The visualization code is available now. You can find it in 'disp.py'. A simple example of how to visualize 
the results is showed in 'eval_planeTR.py'.

## TODO
- ~~Supplement 2D/3D visualization code.~~

## Getting Started

Clone the repository:
```bash
git clone https://github.com/IceTTTb/PlaneTR3D.git
```

We use Python 3.6 and PyTorch 1.6.0 in our implementation, please install dependencies:
```bash
conda create -n planeTR python=3.6
conda activate planeTR
conda install pytorch=1.6.0 torchvision=0.7.0 torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Data Preparation
We train and test our network on the plane dataset created by [PlaneNet](https://github.com/art-programmer/PlaneNet).
We follow [PlaneAE](https://github.com/svip-lab/PlanarReconstruction) to convert the .tfrecords to .npz files.
Please refer to [PlaneAE](https://github.com/svip-lab/PlanarReconstruction) for more details.

We generate line segments using the state-of-the-art line segment detection algorithm [HAWP](https://github.com/cherubicXN/hawp) with their pretrained model.
The processed line segments data we used can be downloaded [here](https://drive.google.com/drive/folders/1Kj6fvTQfPB4TolwiCRpQnlvJfs2vKvJx?usp=sharing).

The structure of the data folder should be
```
plane_data/
  --train/*.npz
  --train_img/*
  --val/*.npz
  --val_img/*
  --train.txt
  --val.txt
```

## Training
Download the pretrained model of [HRNet](https://drive.google.com/drive/folders/1dCq4WxconPEDO8uZq4YDctZ8nxtvEtv8?usp=sharing) 
and place it under the 'ckpts/' folder.

Change the 'root_dir' in config files to the path where you save the data.

Run the following command to train our network on one GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python train_planeTR.py
```
Run the following command to train our network on multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port 295025 train_planeTR.py
```

## Evaluation on ScanNet
Download the pretrained model [here](https://drive.google.com/drive/folders/1dCq4WxconPEDO8uZq4YDctZ8nxtvEtv8?usp=sharing) 
and place it under the 'ckpts/' folder.

Change the 'resume_dir' in 'config_planeTR_eval.yaml' to the path where you save the weight file.

Change the 'root_dir' in config files to the path where you save the data.

Run the following command to evaluate the performance:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_planeTR.py
```

## Evaluation on NYUV2
To evaluate the depth on the NYUV2 dataset, please first download the original data from [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) 
and the official train/test split from [here](http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat). Then, you also 
have to download our line detection results from [here](https://drive.google.com/file/d/1mOY1Hqch0KpfVDC92EUv-Kvh5eBXVIr3/view?usp=sharing). 
After downloading all data, please put them under the folder 'nyudata/'. The structure of the data folder should be:
```
planeTR3D
  --nyudata
    |--nyu_depth_v2_labeled.mat
    |--splits.mat
    |--line_info\
        |--*.txt
```
Run the following command to evaluate the performance:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_nyudepth.py
```

Unfortunately, I lost the original detected line segments used in the paper. Thus I regenerated the line segments on
the NYUV2 dataset with the latest [HAWPV3](https://github.com/cherubicXN/hawp). The depth metrics with these regenerated line segments are:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Rel</th>
<th valign="bottom">log10</th>
<th valign="bottom">RMSE</th>
<th valign="bottom">d1</th>
<th valign="bottom">d2</th>
<th valign="bottom">d3</th>
<tr>
<!-- TABLE BODY -->
<th valign="bottom">0.196</th>
<th valign="bottom">0.096</th>
<th valign="bottom">0.812</th>
<th valign="bottom">63.8</th>
<th valign="bottom">88.3</th>
<th valign="bottom">96.0</th>
</tr>
</tbody></table>
The new results are slightly different to the ones in the paper, but they do not influence the 
conclusion.

## Citations
If you find our work useful in your research, please consider citing:
```
@inproceedings{tan2021planeTR,
title={PlaneTR: Structure-Guided Transformers for 3D Plane Recovery},
author={Tan, Bin and Xue, Nan and Bai, Song and Wu, Tianfu and Xia, Gui-Song},
booktitle = {International Conference on Computer Vision},
year={2021}
}
```

## Contact
[tanbin@whu.edu.cn](tanbin@whu.edu.cn)

[https://xuenan.net/](https://xuenan.net/)

## Acknowledgements
We thank the authors of [PlaneAE](https://github.com/svip-lab/PlanarReconstruction), [PlaneRCNN](https://github.com/NVlabs/planercnn), [interplane](https://github.com/yi-ming-qian/interplane) and [DETR](https://github.com/facebookresearch/detr). Our implementation is heavily built upon their codes.
