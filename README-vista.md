# Environment setup

1. Conda environment

```bash
conda create -n <name> python=3.11

# Assuming in `OLMo/`
pip install -e .

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

2. Bash environment variables

I couldn't remember exactly. But setting this vairables should be useful for `bitsandbytes`, `triton` (which is typically used for LoRA). (Disclaimer: I didn't try to trim the variables to bare minimum)

```bash
# This is vista-specific
export CACHED_PATH_CACHE_ROOT=$WORK/.cache/cached_path
export HF_DATASETS_CACHE=$WORK/.cache/datasets
export HF_HOME=$WORK/.cache/huggingface/hub

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/12.4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/12.4/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/12.4/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/12.4/bin/nvcc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/nvidia_math/12.4/targets/sbsa-linux/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/nvidia_math/12.4/targets/sbsa-linux/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/12.4/targets/sbsa-linux/lib

export CUDA_cublas_LIBRARY=$CUDA_cublas_LIBRARY:/opt/apps/nvidia_math/12.4/targets/sbsa-linux/lib/libcublas.so
export CUDA_cublas_LIBRARY=$CUDA_cublas_LIBRARY:/opt/apps/nvidia_math/12.4/targets/sbsa-linux/lib
```
