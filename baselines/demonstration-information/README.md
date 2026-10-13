# Demonstration Information Estimation

This repository contains code accompanying the paper [Robot Data Curation with Mutual Information Estimators](https://jhejna.github.io/demonstration-info) by Joey Hejna, Suvir Mirchandani, Ashwin Balakrishna, Annie Xie, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, Dhruv Shah, Coline Devin, and Dorsa Sadigh.

This codebase is based off of a frozen version of the [OpenX repository primarily](https://github.com/jhejna/openx) developed by [Joey Hejna](https://jhejna.github.io) for training robot models using Jax, Flax, and RLDS. We build upon ideas used in the [Octo repository](https://github.com/octo-models/octo).

**NOTE: We find that DemInf works best with relative actions. When using your own datasets, we recommend trying relative actions to get stronger results.**

## Installation
First, create a conda environment with python 3.11, and then install requirements and this repo.
```
conda create -n openx python=3.11
pip install -r requirements.txt
pip install -e .
```
If you are on GPU, you will additionally need to install the corresponding jaxlib verison.
```
pip install --upgrade "jax[cuda12_pip]==0.4.37" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
If you are on TPU, instead run:
```
pip install --upgrade "jax[tpu]==0.4.37" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Robomimic**
We use the datasets from the original robomimic paper, e.g. v0.2 which can be found [here](https://robomimic.github.io/docs/v0.2/datasets/robomimic_v0.1.html). Installing the correct robomimic version corresponding to that used in the [original Robomimic paper](https://arxiv.org/abs/2108.03298) is unfortunately a pain. We provide more details commented out in the requirements.txt file, but the basics are as follows.

First, follow the instructions to install `mujoco210_linux` found [here](https://github.com/openai/mujoco-py). This is the original version, not the updated version by GDM.
```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

Then, install robosuite, robomimic, and needed dependencies.
```
# Dependencies
pip install "mujoco-py<2.2,>=2.0"
pip install cython==0.29.37
pip install numba

# Robosuite
git clone https://github.com/ARISE-Initiative/robosuite/
cd robosuite
git checkout offline_study
pip install -e . --no-deps # Ignore
cd ..

# Robomimic
git clone https://github.com/ARISE-Initiative/robomimic/
cd robosuite
git checkout v0.2.0
pip install -e . --no-deps # Ignore
cd ..
```
If compiling `mujoco_py` fails, you may need to change gcc version in the conda environment.

Then repeatedly try to import mujoco_py, robosuite, and robomimic until it works. There are a few manual changes to the code in robosuite and robomimic you will need to make:
1. Comment out all references to EGL Probe if you are using TPU.
2. You will need to change some imports to `from collections.abc` from `from collections`. This is because some typing hints used in robosuite and robomimic were deprecated in Python 3.11.

## Datasets

We use the [RLDS format](https://github.com/google-research/rlds) for datasets following OXE. 

**RoboMimic**
We provide our converter for robomimic in `rlds/robomimic`. You can run it using `tfds build`. Note that as in the instructions, we use the original versions of the robomimic datasets, which can be downloaded [here](https://robomimic.github.io/docs/v0.2/datasets/robomimic_v0.1.html).

**Franka**
We collect data on the Franka robot using the [DROID platform](https://github.com/droid-dataset/droid). We then follow the instructions on [DROID Policy Learning](https://github.com/droid-dataset/droid_policy_learning) for converting the collected demos to RLDS format. Our data can be downloaded [here](https://drive.google.com/file/d/1aVx_2eGp4fKfLkI3h8J1s38lkkXQu4DV/view?usp=sharing).

**RoboCrowd**
Unfortunately, the RoboCrowd datasets are not yet publically available due to the anonymity of crowd-sourced data. We will update this page if/when RoboCrowd data is released.

## Training

**Training Models**

Config files can be found in `configs`. Be sure to replace paths where appropriate for your datasets.

We recommend editting fields at the top of `setup_shell.sh` for your own setup. Then, run `. setup_shell.sh` to setup the training environment. 

Models can then be trained with:

```
python scripts/train.py --config path/to/config:config_str --path save/directory/path --name name/of/run
```
If you want to run to training across multiple datasets, seeds, parameters, baselines etc. we have built in sweeping functionality using any of the `run` scripts in `tools`. For example,
```
python tools/run_local.py --sweep path/to/sweep_file.json --arguments path=globally/set/path project=globally/set/wandb/project ... script specific commands...
```
will print out training commands for all runs. You can use `--save_split=int` to split the jobs across different bash files. Commands for SLURM or TPUs have their own different special flags e.g. `--cpus`, `--mem`, or `--project`.

**Running Quality Inference**

Estimating quality scores for a dataset using different models is done via the `scripts/quality/estimate_quality.py` script. For example,
```
python scripts/quality/estimate_quality.py --estimator=ksg --obs_ckpt=path/to/obs/vae/step --action_ckpt=path/to/action/vae/step --batch_size=1024 --path=path/to/save/scores
```
where the obs_ckpt, action_ckpt, and estimator are set as determined by `scripts/quality/quality_estimators`.

To generate a script that will run all estimators on all models from a sweep, use the following script:
```
python tools/generate_quality_sweep.py --path path/to/sweep/output --output path/to/output/scores --mode image
```
Note that the config names are important for this script, and thus if they change it will not work properly.


**Example Results Pipeline for RoboMimic**

1. Train models by using `tools/run` for one type of launcher, e.g. 
```
python tools/run_local.py --sweep configs/quality/sweep_robomimic_image.json --arguments path=output/robomimic_image --save_split=2

python tools/run_local.py --sweep configs/quality/sweep_robomimic_image_vae.json --arguments path=output/robomimic_image --save_split=2
```

2. Generate inference script and run it
```
python tools/generate_quality_sweep.py --path output/robomimic_image --output output/robomimic_image_inference --mode image > run.sh
bash run.sh
```

3. Generate plots:
```
python scripts/quality/plot.py --path output/robomimic_image_inference --type all --order robomimic_image --title "RoboMimic Image" --use_tf=False --output plot.png
```

