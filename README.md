<div align="center">

<h1><span style="color: #FF8C00;">T</span>abletopGen: <span style="color: #800080;">I</span>nstance-Level Interactive 3D Tabletop Scene Generation from Text or Single Image</h1>

<img src="assets/teaser.png" width="100%" alt="Teaser Image">

<br>

<div style="font-size: 1.5em;">
    <strong>Ziqian Wang</strong><sup>1,3,2*</sup>,
    <strong>Yonghao He</strong><sup>2*†</sup>,
    <strong>Licheng Yang</strong><sup>1,3</sup>,
    <strong>Wei Zou</strong><sup>1,3</sup>,
    <strong>Hongxuan Ma</strong><sup>3</sup>,
    <strong>Liu Liu</strong><sup>4</sup>,
    <br>
    <strong>Wei Sui</strong><sup>2</sup>,
    <strong>Yuxin Guo</strong><sup>1,3</sup>,
    <strong>Hu Su</strong><sup>3✉</sup>
</div>

<br>

<div style="text-align: center;font-size: 1.5em;">
    <sup>1</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences<br>
    <sup>2</sup>D-Robotics<br>
    <sup>3</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS), <br> Institute of Automation, Chinese Academy of Sciences<br>
    <sup>4</sup>Horizon Robotics
</div>

<div style="font-size: 1.5em;font-weight: bold;">
    <sup>*</sup><u>Equal contribution</u> &emsp; <sup>†</sup><u>Project Lead</u> &emsp; <sup>✉</sup><u>Corresponding author</u>
</div>

<br>

<a href="https://arxiv.org/abs/2512.01204"><img src="https://img.shields.io/badge/arXiv-2512.01204-b31b1b.svg" alt="arXiv"></a>
<a href="https://arxiv.org/pdf/2512.01204"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper"></a>
<a href="https://d-robotics-ai-lab.github.io/TabletopGen.project/"><img src="https://img.shields.io/badge/Project-Website-blue.svg" alt="Website"></a>

<br>

<div align="center">
<video src="https://github.com/user-attachments/assets/d266a7bd-308b-4f45-8733-5b5ccf227bc2" controls width="80%"></video>
</div>

</div>

## 🎉 Updates

- **[2025-12-10]** 🎉 TabletopGen is now open source!

## 🧩 Abstract

Generating high-fidelity, physically interactive 3D simulated tabletop scenes is essential for embodied AI—especially for robotic manipulation policy learning and data synthesis. However, current text- or image-driven 3D scene generation methods mainly focus on large-scale scenes, struggling to capture the high-density layouts and complex spatial relations that characterize tabletop scenes. To address these challenges, we propose **TabletopGen**, a training-free, fully automatic framework that generates diverse, instance-level interactive 3D tabletop scenes. TabletopGen accepts a reference image as input, which can be synthesized by a text-to-image model to enhance scene diversity. We then perform instance segmentation and completion on the reference to obtain per-instance images. Each instance is reconstructed into a 3D model followed by canonical coordinate alignment. The aligned 3D models then undergo pose and scale estimation before being assembled into a collision-free, simulation-ready tabletop scene. A key component of our framework is a novel pose and scale alignment approach that decouples the complex spatial reasoning into two stages: a Differentiable Rotation Optimizer for precise rotation recovery and a Top-view Spatial Alignment mechanism for robust translation and scale estimation, enabling accurate 3D reconstruction from 2D reference. Extensive experiments and user studies show that TabletopGen achieves state-of-the-art performance, markedly surpassing existing methods in visual fidelity, layout accuracy, and physical plausibility, capable of generating realistic tabletop scenes with rich stylistic and spatial diversity.


## 🚀 Installation


This project utilizes two distinct environments, **tabletopgen** and **rotation**, to handle complex dependencies.

We provide an automated setup workflow. You **do not** need to manually configure the two environments or compile dependencies one by one.

### 1. Clone the Repository
```bash
git clone https://github.com/D-Robotics-AI-Lab/TabletopGen.git
cd TabletopGen
```

### 2. One-Click Environment Setup
We provide a shell script that automatically:
1.  Creates the primary environment `tabletopgen` (CUDA 11.8, Torch 2.6).
2.  Compiles **Grounded-SAM-2** and installs **BiRefNet**.
3.  Creates the secondary environment `rotation` (CUDA 12.1, PyTorch3D).

**For Linux Users:**
```bash
bash install_env.sh
```
> ☕ **Note:** This process involves compiling CUDA extensions locally. It may take a few minutes depending on your network and CPU.

### 3. Download Model Weights
Run this script to automatically download the correct checkpoints for **BiRefNet**, **SAM 2.1**, and **Grounding DINO** to their respective directories.

```bash
# Activate the main environment first
conda activate tabletopgen

# Run the auto-download script
python install_scripts/download_weights.py
```

## 🛠️ Usage

### 1. Configuration
Before running the pipeline, please configure your API settings (e.g., OpenAI, Hunyuan3D, etc.) in the configuration file:
```yaml
# Edit this file with your own API settings
configs/config.yaml
```

### 2. Generate Input Image (Optional)
If you do not have an input image, you can generate one from text using `text2img.py`.
* **Arguments:**
    * `--doubao_api_key`: Your API key for the generation service.
    * `--text`: Description of the scene (e.g., "A hobby desk with some model cars and tools.").
    * `--id` (Optional): Manually specify the generated image ID. If omitted, it auto-increments.
* **Output:** Generated images will be saved in `scene_image/`.

```bash
conda activate tabletopgen
python text2img.py --doubao_api_key "YOUR_API_KEY" --text "A hobby desk with some model cars and tools."
```

### 3. Run Scene Generation Pipeline
Run the main pipeline to generate the 3D scene.

**Arguments:**
* `--input_image` (Required): Path to the input image file.
* `--scene_id` (Optional): Manually specify the Scene ID (directory name).
* `--skip_step` (Optional): Skip specific pipeline steps (space-separated integers). Useful for debugging or resuming.

**Example Commands:**

```bash
conda activate tabletopgen
python pipeline.py --input_image scene_image/scene_image_1.png

```

> 💡 **Critical Tip for Best Results:**
> In **Step 1** of the pipeline, we **strongly recommend** adjusting the Grounded-SAM-2 thresholds to ensure all object instances are correctly segmented and extracted.
> You can tweak the following parameters in the pipeline code:
> * `box_threshold`
> * `text_threshold`
> * `confidence_threshold`


### 4. Visualization & Simulation
**View GLB Model:**
Once the generation is complete, you can view the assembled 3D scene at:
`output_scene/scene_{id}/scene_{id}.glb`

**NVIDIA Isaac Sim (Physics-based Assembly):**
For a scene assembly with full physical properties, use the Isaac Sim script.
* **Prerequisite:** Ensure NVIDIA Isaac Sim is installed ([Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)).

```bash
# Run the Isaac Sim visualization script
python isaac_final_scene.py
```

## 💝 Acknowledgments


## 📝 Citation

If you use this code in your research, please cite our project:

```bibtex
@article{wang2025tabletopgen,
  title={TabletopGen: Instance-Level Interactive 3D Tabletop Scene Generation from Text or Single Image},
  author={Wang, Ziqian and He, Yonghao and Yang, Licheng and Zou, Wei and Ma, Hongxuan and Liu, Liu and Sui, Wei and Guo, Yuxin and Su, Hu},
  journal={arXiv preprint arXiv:2512.01204},
  year={2025}
}
```
