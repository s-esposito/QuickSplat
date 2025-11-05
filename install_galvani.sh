conda create -n quicksplat python=3.9 -y
conda activate quicksplat
export CUDA_VER=11.8 ; export CUDA_HOME=/usr/local/cuda-$CUDA_VER ; export CUDA_PATH=/usr/local/cuda-$CUDA_VER ; export PATH=/usr/local/cuda-$CUDA_VER/bin:$PATH ; export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VER/lib64:/usr/local/cuda-$CUDA_VER/extras/CUPTI/lib64:$LD_LIBRARY_PATH
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install MinkowskiEngine
conda install openblas-devel -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
# ... follow readme instructions ...

# download and install precompiled PyTorch3D for Py39 CUDA 11.8 and PyTorch 2.2.0
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu118_pyt220.tar.bz2
conda install pytorch3d-0.7.8-py39_cu118_pyt220.tar.bz2

pip install -r requirements.txt
# Install additional dependencies
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-surfel-rasterization
