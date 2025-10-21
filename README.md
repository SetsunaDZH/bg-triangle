# BG-Triangle: Bézier Gaussian Triangle for 3D Vectorization and Rendering [[Paper]](https://www.arxiv.org/pdf/2503.13961) [[Project Page]](https://wuminye.github.io/projects/BGTriangle/) 
Minye Wu*, Haizhao Dai*, Kaixin Yao,  Tinne Tuytelaars#, and Jingyi Yu#.</br>


![framework](teaser.png)

> 📘 **细分曲面替换说明**：项目已经将 Bézier 三角面实现替换为细分曲面评估器，详细的设计动机与接口兼容性说明见 [`docs/subdivision_surface_overview.md`](docs/subdivision_surface_overview.md)。

## Abstract
Differentiable rendering enables efficient optimization by allowing gradients to be computed through the rendering process, facilitating 3D reconstruction, inverse rendering and neural scene representation learning. To ensure differentiability, existing solutions approximate or re-formulate traditional rendering operations using smooth, probabilistic proxies such as volumes or Gaussian primitives. Consequently, they struggle to preserve sharp edges due to the lack of explicit boundary definitions. We present a novel hybrid representation, Bézier Gaussian Triangle (BG-Triangle), that combines Bézier triangle-based vector graphics primitives with Gaussian-based probabilistic models, to maintain accurate shape modeling while conducting resolution-independent differentiable rendering. We present a robust and effective discontinuity-aware rendering technique to reduce uncertainties at object boundaries. We also employ an adaptive densification and pruning scheme for efficient training while reliably handling level-of-detail (LoD) variations. Experiments show that BG-Triangle achieves comparable rendering quality as 3DGS but with superior boundary preservation. More importantly, BG-Triangle uses a much smaller number of primitives than its alternatives, showcasing the benefits of vectorized graphics primitives and the potential to bridge the gap between classic and emerging representations.




[![Watch the video](https://img.youtube.com/vi/D56aqZA8LKw/0.jpg)](https://www.youtube.com/watch?v=D56aqZA8LKw)





## Get Started

### 0.Software Requirements
- Conda 
- C++ Compiler for PyTorch extensions
- CUDA SDK 11 for PyTorch extensions (version 11.8)
- C++ Compiler and CUDA SDK must be compatible


### 1.Setup
Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate bgtriangle
```

Please do **NOT** forget to compile the CUDA kernels:
```shell
pip install submodules/brasterizer/
pip install submodules/diff-Brasterization/
```


### 2.Training

To run the optimizer, simply use
```
python demo.py  -s <path to COLMAP or NeRF Synthetic dataset>  --init_path <path to initial point cloud ply file>  --model_path <path to output folder>  --eval --disable_viewer
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```0.0.0.0``` by default.
  #### --port 
  Port to use for GUI server, ```8080``` by default.
  #### --disable_viewer
  Disable viewer during training
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.

</details>
<br>

For [Masked T&T dataset](https://drive.google.com/file/d/1rnJPlxkK_rj5pNtv5BTtfkyMXHv-IbZT/view?usp=sharing), please use the provided training script (train_maskTT.sh).


### 3.Testing and Evaluation
You can render training/test sets and produce error metrics as follows:
```shell
python test.py  -s <path to COLMAP or NeRF Synthetic dataset> --model_path <path to trained model> --checkpoint <path to checkpoint file>  --eval --log_blur_radius -9 --mode 2
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

### 4.Viewer
We also provide a viewer for training and testing visualization via browsers.

If you would like to enable the training-time viewer, please do not pass '--disable_viewer' parameters to the training code. Once you have a trained model, you can visualizate it by using this command:
```
python viewer.py  -s <path to COLMAP or NeRF Synthetic dataset> --model_path <path to trained model> --start_checkpoint <path to checkpoint file> --log_blur_radius -9
```

## Initial Point Clouds
We provide the initial [point clouds](initial_pc/nerf_synthetic.zip) for NeRF Synthetic dataset, which are used in training. 


## Licenses

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All material is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.


