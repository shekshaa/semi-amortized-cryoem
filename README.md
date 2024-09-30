# Cryo-SAPIENCE | NeurIPS (2024)

### [Project Page]() | [Arxiv](https://arxiv.org/abs/2406.10455)
This is the official PyTorch implementation of the paper [Improving Ab-Initio Cryo-EM Reconstruction with Semi-Amortized Pose Inference](https://arxiv.org/abs/2406.10455).
We develop a new approach to Ab-initio <b><span style="font-size: 1.1em;">Cryo</span></b>-EM Reconstruction with 
<b><span style="font-size: 1.1em;">S</span></b>emi-<b><span style="font-size: 1.1em;">A</span></b>mortized <b><span style="font-size: 1.1em;">P</span></b>ose <b><span style="font-size: 1.1em;">I</span></b>nfer<b><span style="font-size: 1.1em;">ENCE</span></b> (Cryo-SAPIENCE).
<!-- **S**emi-**A**mortized **P**ose **I**nfer**ence**. -->
<img src='./media/teaser.png'/>
<br>

## Dependencies
The code is tested on Python `3.9` and Pytorch `1.12` with cuda version `11.3`.
Please run following commands to create a compatible (mini)conda environment called `semi-amortized`:
```
# create env
conda create -n semi-amortized python=3.9
conda activate semi-amortized

# pytorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# other libraries
pip install scipy==1.9.3 scikit-image==0.22.0 tqdm PyYAML matplotlib kornia notebook tensorboard numpy==1.23.4
pip install starfile==0.4.5 mrcfile
```
The code also depends on Pytorch3D. Install the latest stable version of it by running:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
We also use [EMAN2](https://cryoem.bcm.edu/cryoem/downloads/view_eman2_versions) command line interface to align the predicted map with the ground truth before computing Fourier Shell Correlation (FSC) and resolution.

## Synthetic Data
To reproduce results, you first need to syntheticly generate 2D projections based on given 3D density maps.
You can find density maps of Spliceosome, Spike Protein, and Heat Shock Protein (HSP) stored in `mrcfiles` folder.
For each dataset, we provide a config file which primarily defines path to density map (`.mrc`), the number projections, and image size (e.g. `128`). 
See the config file for more parameters.
To generate data, run `generate_data.py` with the corresponding config file, e.g. for HSP:
```
python generate_data.py --config ./configs/mrc2star_hsp.yaml
```
As a result, the dataset will get stored locally in `./synthetic_data/hsp/`. 
In this folder, you can find a star file called `data.star` storing the metadata (such as CTF parameters) accompanied with a folder called `Particles` storing particle images into several `.mrcs` files.

Once the synthetic data is ready, you can run the semi-amortized method,
```
python train_semi-amortized.py --config ./configs/train_synth.yaml --save_path path/to/save/logs
```
which will write the reconstruction logs in `path/to/save/logs`. You can run `tensorboard` to see several curves such as reconstruction error and mean/median rotation errors.

Within the config file `train.yaml`, several hyperparameters are defined. An important one is `num_rotations` which determines number of heads of CNN encoder during auto-encoding stage.
Moreover, `epochs_amortized` and `epochs_unamortized` specify number of epochs spent on auto-encoding and auto-decoding stages, respectively.
We provide a brief description for each hyperparamter in the config file.

## Experimental Data
To evaluate on real datasets, we use 80S Ribosome ([EMPIAR10028](https://www.ebi.ac.uk/empiar/EMPIAR-10028/)).
Particle images are originally of size `D=360` with `Apix=1.34A`.
We downsample them to `D=128`. 
To ensure reproduciblity, we provide the corresponding particles and metadata in [Zenodo link](https://zenodo.org/records/13863054?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijg3MTFiNTcxLTgwMmMtNDYxZC04YWU5LWFjYTQyYjZhNDEyZiIsImRhdGEiOnt9LCJyYW5kb20iOiJiOWQ5NmE3MTNkM2ZmOWZlZmQwMWE0Yzg2OGZhNjE5MCJ9.U71prW_3uD374_p1hrGJYIZb4t0pihK8yPhVcj4xma2tO6qPq4BzZwe9HIqtzrium2a54tiVCQpqDWOs407yZg). Please download the data and place it in `real_data` folder. 

Once the data is ready, you can run the semi-amortized method using the config file `train_real.yaml`:
```
python train_semi-amortized.py --config ./configs/train_real.yaml --save_path path/to/save/logs
```
Similarly, the results will be saved in `path/to/save/logs`.

## Citation
```
@article{shekarforoush2024improving,
  title={Improving Ab-Initio Cryo-EM Reconstruction with Semi-Amortized Pose Inference},
  author={Shekarforoush, Shayan and Lindell, David B and Brubaker, Marcus A and Fleet, David J},
  journal={arXiv preprint arXiv:2406.10455},
  year={2024}
}
```

## Acknowledgement
This research was supported in part by the Province of Ontario, the Government of Canada, through NSERC, CIFAR, and the Canada First Research Excellence Fund for the Vision, Science to Applications (VISTA) programme, and by companies sponsoring the Vector Institute.

<!-- ## Analysis and Visualizations -->

<!-- As a simple toy example, run the following command to start a slurm job which uses the semi-amortized method for 3D reconstruction based on a synthetic dataset of spliceosome. Basic configs are stored in `./configs/base.yaml`.
```
sbatch slurm_semi_amortized.sh "./configs/base.yaml"
```

This job stores the results in `/checkpoint/job_id/` where `job_id` is a unqiue number identifying the job. You can run `tensorboard` on this folder:
```
tensorboard --logdir /checkpoint/job_id --host 0.0.0.0 --port 6006
```
To see results locally, use ssh to port-forward from the node (where tensorboard is running) on port 6006 to your localhost. -->