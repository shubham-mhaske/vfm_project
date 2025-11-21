# Visual Foundation Model for Histopathology Image Segmentation

This project provides a comprehensive framework for utilizing and evaluating Visual Foundation Models (VFMs) like the Segment Anything Model (SAM/SAM2) for medical image segmentation, with a specific focus on breast cancer histology images from the BCSS dataset.

The framework supports various prompting strategies for zero-shot segmentation, including hard-coded prompts and LLM-generated text and multimodal prompts. It also includes scripts for fine-tuning SAM, evaluating segmentation performance, and running experiments in a reproducible manner, both locally and on High-Performance Computing (HPC) clusters.

## Features

- **Zero-Shot Segmentation**: Evaluate pre-trained SAM with different prompting strategies.
- **LLM-Powered Prompt Engineering**: Automatically generate text and multimodal prompts using Large Language Models (LLMs) like Gemini.
- **Fine-Tuning**: Fine-tune the SAM model on the BCSS dataset for improved performance.
- **Comprehensive Evaluation**: Calculate segmentation metrics (like Dice score) and class-based performance.
- **Reproducible Experiments**: Standardized scripts and configuration files for running experiments.
- **HPC Ready**: Includes a SLURM script for running jobs on GPU nodes.

## Directory Structure

```
/
├───conf/                     # Hydra configuration directory for experiments.
│   └───experiment/           # Configs for different fine-tuning experiments.
├───data/                     # Root directory for datasets (not tracked by Git).
│   └───bcss/                 # BCSS dataset images and masks.
├───finetune_logs/            # Output directory for fine-tuning runs.
├───src/                      # All Python source code.
│   ├───main.py               # Main script for zero-shot experiments.
│   ├───run_finetuning.py     # Script for launching fine-tuning experiments.
│   ├───sam_segmentation.py   # Core logic for SAM-based segmentation.
│   ├───evaluation.py         # Logic for evaluating segmentation masks.
│   ├───dataset.py            # PyTorch Dataset and DataLoader definitions.
│   └───...
├───sam2/                     # Git submodule for the SAM2 model.
├───...
└───README.md                 # This file.
```

## Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the Main Repository**
```bash
git clone <your-repository-url>
cd <project-directory>
```

**2. Set Up Nested Submodules**
This project uses Git submodules for `sam2` and the dataset downloader. Initialize them with:
```bash
git submodule update --init --recursive
```

**3. Create a Python Environment and Install Dependencies**
It is recommended to use a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. Download Pre-trained Models**
- **SAM2 Models**: The `sam2` submodule includes a script to download its models.
  ```bash
  bash sam2/checkpoints/download_ckpts.sh
  ```
- **Original SAM Model**: Download the official ViT-H SAM model.
  ```bash
  mkdir -p models/sam_checkpoints
  wget -P models/sam_checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  ```

**5. Download the Dataset**
The dataset is downloaded using the script from the `CrowdsourcingDataset-Amgadetal2019` submodule.
```bash
# Navigate to the submodule directory
cd CrowdsourcingDataset-Amgadetal2019

# Install its specific dependencies
pip install girder_client numpy imageio scikit-image

# Run the download script (you may need to register for an account)
python download_crowdsource_dataset.py

# Return to the project root
cd ..

# Organize the data into the project's data directory
# Create the destination folders
mkdir -p data/bcss/images data/bcss/masks

# Move the downloaded files
mv CrowdsourcingDataset-Amgadetal2019/data/wsis/* data/bcss/images/
mv CrowdsourcingDataset-Amgadetal2019/data/annotations/* data/bcss/masks/
```

## Running Experiments

This project supports two main types of experiments: zero-shot evaluation and model fine-tuning.

### Zero-Shot Evaluation
Zero-shot experiments are run via `src/main.py` and use pre-trained models to perform segmentation based on different prompting strategies. The prompts themselves are configured in `configs/prompts/`.

**Example Usage:**
```bash
# Run end-to-end segmentation and classification using a prompt file
python src/main.py --prompt_file configs/prompts/llm_text_prompts_v2_clip_friendly.json
```

### Fine-Tuning
Fine-tuning experiments are managed by Hydra and the new `src/run_finetuning.py` script. This provides a powerful and organized way to define and track experiments.

**Configuration Structure**
- `conf/config.yaml`: The main Hydra config file.
- `conf/experiment/`: This directory contains YAML files for each experiment.
- `conf/experiment/base_finetune.yaml`: A base configuration with shared settings. Individual experiments inherit from this and override specific parameters.

**Running an Experiment**
You can run an experiment by specifying its name. Hydra will automatically create a timestamped output directory in `finetune_logs/` containing the full config, logs, and checkpoints for the run.

```bash
# Run the base fine-tuning experiment
conda run -n vfm_project python src/run_finetuning.py

# Run a specific experiment (defined in conf/experiment/low_lr.yaml)
conda run -n vfm_project python src/run_finetuning.py experiment=low_lr

# Override any parameter from the command line
conda run -n vfm_project python src/run_finetuning.py experiment=strong_aug scratch.train_batch_size=4
```

**Creating a New Experiment**
To create a new experiment, simply add a new YAML file in `conf/experiment/`. For example, `my_experiment.yaml`:
```yaml
# conf/experiment/my_experiment.yaml
defaults:
  - base_finetune

experiment:
  name: my_experiment

# Override parameters for this experiment
scratch:
  num_epochs: 100
```
Run it with `python src/run_finetuning.py experiment=my_experiment`.


## Running on an HPC Cluster (SLURM)

You can easily run the new fine-tuning framework on a Slurm cluster. Create a script like `run_finetuning.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=sam_finetune
#SBATCH --output=slurm_logs/sam_finetune_%j.out
#SBATCH --error=slurm_logs/sam_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00

# Activate your environment
module restore dl
source $SCRATCH/vfm_env/bin/activate
cd $SCRATCH/vfm_project

# Run the desired experiment
# The output directory will be managed by Hydra inside 'finetune_logs/'
echo "Starting SAM2 finetuning..."
python src/run_finetuning.py experiment=base_finetune
echo "Finetuning completed."
```
Submit the job to the queue with `sbatch run_finetuning.slurm`. You can easily change the experiment by modifying the `python` command in the script.


## Dataset Details

This project uses the Breast Cancer Semantic Segmentation (BCSS) dataset from Amgad et al., 2019.

- **Classes**: The masks contain integer pixel values corresponding to different tissue types:
  - `0`: Background
  - `1`: Tumor
  - `2`: Stroma
  - `3`: Lymphocyte
  - `4`: Necrosis
  - `5`: Blood Vessel
- **Data Split**: The `src/dataset.py` script performs a deterministic split into training, validation, and test sets based on the TCGA slide identifiers to ensure that images from the same patient do not cross from the training set into the test set.