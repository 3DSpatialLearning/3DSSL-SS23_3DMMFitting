## Installation

### Windows/Linux
```bash
pip install -e .
pip install -r requirements.txt
```
Install pytorch3d following the steps from: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

### MacOS
```bash
pip install -e .
pip install -r requirements-mac.txt
```
## Conda environment setup
```bash
conda create -n 3d_fitting python=3.9 -y
conda activate 3d_fitting

# Install dependencies
pip install -e .
pip install -r requirements.txt

# Install pytorch with conda
## First uninstall the installed torch
pip uninstall torch -y

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 cudatoolkit=11.7 -c pytorch -c nvidia


# Install pytroch 3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# Compile the nms module of PIPnet
cd models/PIPNET/FaceBoxesV2/utils
sh make.sh

# Some package got overwritten, we need to uninstall them first
pip uninstall numpy
pip install numpy==1.23.1

# export CUDA_HOME path
