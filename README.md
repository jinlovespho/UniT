<div align="center">
<h1>
UniT: Unified Diffusion Transformer for High-fidelity Text-Aware Image Restoration</h1>


[**Jin Hyeon Kim**](https://github.com/jinlovespho)<sup>1</sup>,
**Paul Hyunbin Cho**<sup>1</sup>, 
**Claire Kim**<sup>1</sup>,

[**Jaewon Min**](https://github.com/Min-Jaewon/)<sup>1</sup>, 
[**Jaeeun Lee**](https://github.com/babywhale03)<sup>2</sup>,
**Jihye Park**<sup>2</sup>, **Yeji Choi**<sup>1</sup>, [**Seungryong Kim**](https://scholar.google.com/citations?hl=zh-CN&user=cIK1hS8AAAAJ)<sup>1&dagger;</sup>

<sup>1</sup> KAIST&nbsp;AI ·
<sup>2</sup> Samsung&nbsp;Electronics
<sup>&dagger;</sup>Corresponding Author.

<a href="https://arxiv.org/abs/2506.09993"><img src="https://img.shields.io/badge/arXiv-2506.09993-B31B1B"></a>
        <a href="https://cvlab-kaist.github.io/TAIR/"><img src="https://img.shields.io/badge/Project%20Page-online-1E90FF"></a>
</div>


# 🔈 News 
- [ ] Training code
- [ ] Inference code 
- [x] 🚀 **2025.12.02** — Official launch of the repository and project page!




#  👟 Installation Walkthrough


### 1. Conda Environment

```bash
conda create -n unit python=3.9 -y
conda activate unit
```

**2. Library Installation**
- Download the following libraries in the order listed below.
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install numpy==1.26.3 --no-deps
pip install pyiqa==0.1.14.1 --no-deps 
cd detectron2 
pip install -e . --no-deps --no-build-isolation --config-settings editable_mode=compat
cd ../testr 
pip install -e . --no-deps --no-build-isolation --config-settings editable_mode=compat
pip install cloudpickle --no-deps
```

### 3. Model Weights
  Download the respective model weights by running the respective bash files below.


- **SD3 weights**
```bash
# First login enter your HF access token to download SD3 weights
huggingface-cli login   

# Then run the bash file
bash download_bash/download_sd3.sh 
```

- **DiT4SR weights**
```bash
bash download_bash/download_dit4sr.sh 
```

- **TESTR weights**
```bash
bash download_bash/download_testr.sh 
```


## 🚀 Inference Demo
### Demo Script 
Download the released checkpoint of our model (UniT) from [google drive](), and set the corresponding path in the demo [configuration file](). Then, run the script below to perform text-aware image restoration on low-quality image samples. The results will be saved in val_demo_result/ by default.
```bash

```


### Demo Result
Running the demo inference script will generate the following text restoration results. The visualized images are shown in the order: Low-Quality (LQ) image / Restored image / High-Quality (HQ) Ground Truth image. Note that when the text in the LQ images is severely degraded, the model may fail to accurately restore the textual content due to insufficient visual information.visualized




## 🔥 Training Recipe  
Before training, modify the training script and training config.
- training script -> for setting cuda and GPU
- training config -> for setting data_path, training_batch_size, etc

### Stage1 Training

 - Stage1 training script: [JIHYE_train_stage1_dit4sr.sh](run_scripts/train/JIHYE_train_stage1_dit4sr.sh)
 - Stage1 training config: [JIHYE_train_stage1_dit4sr.yaml](run_configs/train/JIHYE_train_stage1_dit4sr.yaml)

```bash
# stage1 training 
bash run_scripts/train/JIHYE_train_stage1_dit4sr.sh
```

### Stage2 Training
- Stage2 training script: [JIHYE_train_stage2_testr.sh](run_scripts/train/JIHYE_train_stage2_testr.sh)
- Stage2 training config: [JIHYE_train_stage2_testr.yaml](run_configs/train/JIHYE_train_stage2_testr.yaml)

```bash
# stage2 training 
bash run_scripts/train/JIHYE_train_stage2_testr.sh
```

 ### Stage3 Training
 - Stage3 training script: [JIHYE_train_stage3_dit4sr_testr.sh](run_scripts/train/JIHYE_train_stage3_dit4sr_testr.sh)
 - Stage3 training config: [JIHYE_train_stage3_dit4sr_testr.yaml](run_configs/train/JIHYE_train_stage3_dit4sr_testr.yaml)

```bash
# stage3 training 
bash run_scripts/train/JIHYE_train_stage3_dit4sr_testr.sh
```
