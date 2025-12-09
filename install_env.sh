#!/bin/bash

# Scripts for installing TableTopGen and Rotation environments
set -e

echo "========================================================="
echo "   Start Installing TableTopGen & Rotation Environments"
echo "========================================================="

# 1. Initialize Conda
eval "$(conda shell.bash hook)"

# ==========================================
# Part 1: Install TableTopGen (main environment)
# ==========================================
echo "[1/2] Creating environment: tabletopgen..."
conda create -n tabletopgen python=3.10 -y
conda activate tabletopgen

echo "Installing PyTorch for tabletopgen (CUDA 11.8)..."

pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118

echo "Compiling Grounded-SAM-2..."
cd Grounded-SAM-2
export CUDA_HOME=/usr/local/cuda-11.8 
pip install --no-build-isolation -e .

echo "Installing Grounding DINO..."

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
conda install ninja -y
conda deactivate
conda activate tabletopgen

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

retry_pip_editable() {
    local pkg_path="$1"
    echo "Trying to install editable package: $pkg_path"
    if pip install --no-build-isolation -e "$pkg_path"; then
        echo "Install succeeded: $pkg_path"
        return 0
    fi

    echo "First attempt failed, retrying once..."
    if pip install --no-build-isolation -e "$pkg_path"; then
        echo "Install succeeded on retry: $pkg_path"
        return 0
    else
        echo "Install failed twice: $pkg_path"
        return 1
    fi
}

retry_pip_editable grounding_dino

cd ..

echo "Installing BiRefNet dependencies..."
cd BiRefNet
pip install -r requirements.txt
cd ..

echo "Installing Dependencies..."
pip install -r install_scripts/requirements_tabletop.txt

echo "TableTopGen Installed Successfully!"

# ==========================================
# Part 2: Install Rotation (sub-environment)
# ==========================================
echo "[2/2] Creating environment: rotation..."
conda deactivate
conda create -n rotation python=3.10 -y
conda activate rotation

echo "Installing PyTorch for rotation (CUDA 12.1)..."

pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu121

echo "Installing PyTorch3D and others..."
pip install pytorch3d==0.7.8 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
pip install -r install_scripts/requirements_rotation.txt

echo "========================================================="
echo "   All Environments Installed Successfully!"
echo "   Use 'conda activate tabletopgen' to start."
echo "========================================================="
