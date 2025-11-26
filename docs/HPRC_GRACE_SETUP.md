# HPRC Grace Setup Guide

Guide for running deep learning training on Texas A&M's HPRC Grace cluster.

## Table of Contents
1. [Access & VPN Setup](#access--vpn-setup)
2. [Environment Setup](#environment-setup)
3. [Project Setup](#project-setup)
4. [Running Jobs](#running-jobs)
5. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
6. [Quick Reference](#quick-reference)

---

## Access & VPN Setup

### 1. Request HPRC Access

1. Visit [hprc.tamu.edu/apply](https://hprc.tamu.edu/apply/)
2. Log in with NetID and request **Grace** cluster access
3. Wait for approval (1-2 business days)

### 2. Install VPN (Off-Campus Only)

**VPN Required:** Off-campus access requires TAMU VPN. On-campus networks (eduroam, TAMU wireless) don't need VPN.

**Install:**
- Download from [connect.tamu.edu](https://connect.tamu.edu)
- Or via Homebrew (macOS): `brew install --cask cisco-anyconnect`
- Or OpenConnect (Linux): `sudo apt-get install openconnect`

**Connect:**
```bash
# Cisco AnyConnect GUI: connect to connect.tamu.edu
# Or via CLI:
sudo openconnect connect.tamu.edu
# Enter NetID + password, complete Duo authentication

# Verify connection
ping grace.hprc.tamu.edu
```

### 3. SSH to Grace

```bash
ssh your_netid@grace.hprc.tamu.edu

# Setup SSH key (optional, for password-less login)
ssh-keygen -t rsa -b 4096
ssh-copy-id your_netid@grace.hprc.tamu.edu
```

**Troubleshooting:** If connection fails, verify VPN is connected, check [hprc.tamu.edu/status](https://hprc.tamu.edu/status/), and ensure account is approved.

### 4. Understand File Systems

| Path | Purpose | Quota | Backed Up |
|------|---------|-------|-----------|
| `$HOME` | Scripts, configs | 10 GB | Yes |
| `$SCRATCH` | Training, datasets | 1 TB | No |
| `$WORK` | Long-term storage | Varies | Yes |

**→ Use `$SCRATCH` for all training work** (optimized for high I/O)

---

## Environment Setup

### 1. Setup Modules

Create a saved collection of required modules:

```bash
module purge
module load GCCcore/11.3.0 Python/3.10.4-GCCcore-11.3.0 CUDA/11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0
module save dl  # Save as "dl" collection

# Restore in future sessions
module restore dl
```

### 2. Create Virtual Environment

```bash
cd $SCRATCH
python3 -m venv vfm_env
source vfm_env/bin/activate
pip install --upgrade pip
```

### 3. Install PyTorch

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Should show: 2.0.1 True
```

---

## Project Setup

### 1. Clone & Install

```bash
cd $SCRATCH
git clone https://github.com/shubham-mhaske/vfm_project.git
cd vfm_project
git submodule update --init --recursive
pip install -r requirements.txt
```

### 2. Download Checkpoints

```bash
bash sam2/checkpoints/download_ckpts.sh
ls -lh sam2/checkpoints/sam2.1_hiera_large.pt  # Verify ~857 MB
```

### 3. Prepare Dataset

```bash
mkdir -p data/bcss/{images,masks}

# Transfer from local machine:
rsync -avzP data/bcss/ your_netid@grace.hprc.tamu.edu:$SCRATCH/vfm_project/data/bcss/

# Verify
ls data/bcss/images/*.png | wc -l  # Should be 151
```

### 4. Create Directories

```bash
mkdir -p logs finetune_logs results_segmentation
```

---

## Module Management

### Understanding Modules on Grace

Modules are the way Grace manages different software versions and dependencies.

#### Common Module Commands

```bash
# List all available modules
module avail

# Search for specific modules
module avail python
module avail cuda

# Show currently loaded modules
module list

# Load a module
module load Python/3.10.4-GCCcore-11.3.0

# Unload a module
module unload Python

# Purge all modules (start clean)
module purge

# Show details about a module
module show CUDA/11.7.0

# List saved collections
module savelist

# Restore a saved collection
module restore dl
```

### Required Modules for This Project

Create a module collection with these modules:

```bash
module purge
module load GCCcore/11.3.0
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load git/2.36.1-GCCcore-11.3.0-nodocs
module save dl
```

**Why these modules?**
- **GCCcore**: Base compiler toolchain
- **Python**: Python interpreter (3.10 is compatible with PyTorch 2.0)
- **CUDA**: GPU acceleration library
- **cuDNN**: Deep learning GPU acceleration
- **git**: Version control

---

## Running Training Jobs

### 1. Understanding SLURM

Grace uses SLURM (Simple Linux Utility for Resource Management) as its job scheduler. You don't run jobs directly on login nodes - you submit them to SLURM.

#### Key SLURM Concepts

- **Login Node**: Where you SSH in. For editing files and submitting jobs only.
- **Compute Node**: Where jobs actually run. These have GPUs.
- **Partition**: Group of compute nodes (`gpu` partition for GPU jobs)
- **Job Script**: Bash script with SLURM directives (`#SBATCH`)

### 2. SLURM Script Anatomy

The SLURM job script `scripts/slurm/run_training_grace.slurm` has three sections:

#### A. SLURM Directives (Resource Requests)

```bash
#!/bin/bash
#SBATCH --job-name=sam_finetune_best       # Job name (shows in queue)
#SBATCH --output=logs/sam_finetune_%j.out   # stdout log (%j = job ID)
#SBATCH --error=logs/sam_finetune_%j.err    # stderr log
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                 # MPI tasks per node (1 for single-GPU)
#SBATCH --cpus-per-task=16                  # CPU cores (for data loading)
#SBATCH --mem=128G                          # RAM per node
#SBATCH --gres=gpu:a100:1                   # GPU type and count (1 A100)
#SBATCH --partition=gpu                     # Which partition/queue
#SBATCH --time=1-00:00:00                   # Max runtime (1 day)
```

**Resource Guidelines:**
- **For training 85 images:** 1 A100, 16 CPUs, 128GB RAM is optimal
- **For inference/evaluation:** 1 A100, 32 CPUs, 128GB RAM
- **Time limit format:** `DD-HH:MM:SS` (e.g., `1-00:00:00` = 1 day)

#### B. Environment Setup

```bash
# Start clean
module purge

# Load saved deep learning modules
module restore dl

# Activate Python environment
source $SCRATCH/vfm_env/bin/activate

# Fix library path issues
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Navigate to project
cd $SCRATCH/vfm_project

# Set Python path for imports
export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"
```

**Critical:** The `PYTHONPATH` export ensures Python can find your project modules and the SAM2 submodule.

#### C. Job Execution

```bash
# Create log directory if it doesn't exist
mkdir -p logs

# Run the training script
echo "Starting SAM2 finetuning..."
python src/run_finetuning.py experiment=base_finetune
echo "Finetuning completed."
```

### 3. Submitting a Job

```bash
# Make sure you're in the project directory
cd $SCRATCH/vfm_project

# Submit the job
sbatch scripts/slurm/run_training_grace.slurm

# Output will show:
# Submitted batch job 123456
```

### 4. Common SLURM Job Patterns

#### Dry Run (Quick Test)

Create `test_job.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00  # 30 minutes

module restore dl
source $SCRATCH/vfm_env/bin/activate
cd $SCRATCH/vfm_project
export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"

# Test with 1 epoch only
python src/run_finetuning.py experiment=base_finetune scratch.num_epochs=1
```

#### Evaluation Job

Use `scripts/slurm/evaluate_segmentation.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=eval_segmentation
#SBATCH --output=results_segmentation/eval_%j.log
#SBATCH --error=results_segmentation/eval_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=1-12:00:00

module restore dl
source $SCRATCH/vfm_env/bin/activate
cd $SCRATCH/vfm_project
export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"

python -u src/evaluate_segmentation.py \
  --checkpoint finetune_logs/checkpoints/checkpoint_82.pt \
  --model_cfg sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
  --image_dir data/bcss/images \
  --mask_dir data/bcss/masks \
  --split test \
  --prompt_type box \
  --use_neg_points \
  --output_dir results_segmentation \
  --verbose
```

---

## Monitoring Jobs

### 1. Check Job Queue Status

```bash
# View your jobs
squeue -u $USER

# Detailed view
squeue -u $USER -l

# Watch continuously (updates every 30 seconds)
watch -n 30 'squeue -u $USER'
```

**Job Status Codes:**
- `PD` (Pending): Waiting in queue
- `R` (Running): Currently executing
- `CG` (Completing): Finishing up
- `CD` (Completed): Finished successfully
- `F` (Failed): Job failed

### 2. Monitor Job Output in Real-Time

```bash
# Once job starts, tail the output log
tail -f logs/sam_finetune_<jobid>.out

# Example with actual job ID
tail -f logs/sam_finetune_123456.out

# View error log
tail -f logs/sam_finetune_123456.err
```

### 3. Check Job Details

```bash
# Show job info
scontrol show job 123456

# View job efficiency after completion
seff 123456
```

### 4. Monitor GPU Usage (If Job is Running)

```bash
# SSH into the compute node where your job is running
# First, find which node:
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Then SSH to that node (e.g., gpu001)
ssh gpu001

# Check GPU usage
nvidia-smi

# Watch GPU usage
watch -n 1 nvidia-smi
```

### 5. Cancel a Job

```bash
# Cancel specific job
scancel 123456

# Cancel all your jobs
scancel -u $USER

# Cancel all your jobs with a specific name
scancel -u $USER -n sam_finetune_best
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Job Stays in Pending (PD) State

**Symptoms:**
```bash
$ squeue -u $USER
  JOBID     NAME     ST   TIME NODES
  123456    sam_fi   PD   0:00     1
```

**Possible Causes:**
- Requested resources not available (e.g., all A100s in use)
- Time limit too long for partition
- Invalid resource request

**Solutions:**
```bash
# Check why job is pending
squeue -u $USER --start

# Try requesting fewer resources
#SBATCH --gres=gpu:1  # Instead of gpu:a100:1 (any available GPU)

# Or reduce time limit
#SBATCH --time=12:00:00  # 12 hours instead of 1 day
```

#### Issue 2: Job Fails Immediately

**Symptoms:**
```bash
$ squeue -u $USER
# Job disappears immediately after submission
```

**Diagnosis:**
```bash
# Check error log
cat logs/sam_finetune_123456.err

# Common errors and fixes:

# Error: "ModuleNotFoundError: No module named 'torch'"
# Fix: Ensure virtual environment is activated in SLURM script
source $SCRATCH/vfm_env/bin/activate

# Error: "CUDA out of memory"
# Fix: Reduce batch size in config
python src/run_finetuning.py experiment=base_finetune scratch.train_batch_size=2

# Error: "No such file or directory: sam2/checkpoints/..."
# Fix: Download checkpoints
bash sam2/checkpoints/download_ckpts.sh
```

#### Issue 3: Module Load Failures

**Symptoms:**
```
Lmod has detected the following error: The following module(s) are unknown: "Python/3.10.4-GCCcore-11.3.0"
```

**Solution:**
```bash
# Check available Python modules
module avail python

# Use exact module name from output
module load Python/3.10.4  # Adjust to available version

# Update your saved collection
module purge
module load [correct modules]
module save dl
```

#### Issue 4: Python Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'dataset'
ImportError: cannot import name 'BCSSDataset'
```

**Solution:**
```bash
# Ensure PYTHONPATH is set correctly in SLURM script
export PYTHONPATH="$SCRATCH/vfm_project:$SCRATCH/vfm_project/sam2:$PYTHONPATH"

# Verify in an interactive job:
python -c "import sys; print('\n'.join(sys.path))"
```

#### Issue 5: Disk Quota Exceeded

**Symptoms:**
```
OSError: [Errno 122] Disk quota exceeded
```

**Solution:**
```bash
# Check disk usage
df -h $SCRATCH
quota -s

# Clean up old logs and checkpoints
cd $SCRATCH/vfm_project
rm -rf finetune_logs/dry_run-*
rm -rf results_segmentation/old_results

# If HOME is full, check and clean:
du -sh $HOME/*
```

#### Issue 6: Checkpoint Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'sam2/checkpoints/sam2.1_hiera_large.pt'
```

**Solution:**
```bash
cd $SCRATCH/vfm_project
bash sam2/checkpoints/download_ckpts.sh

# Verify
ls -lh sam2/checkpoints/*.pt
```

---

## Best Practices

### 1. Resource Allocation

**Start Small, Scale Up:**
```bash
# Test run (1 epoch, small time)
#SBATCH --time=00:30:00
python src/run_finetuning.py experiment=base_finetune scratch.num_epochs=1

# If successful, full run
#SBATCH --time=1-00:00:00
python src/run_finetuning.py experiment=base_finetune
```

**GPU Selection:**
- **A100 (40GB)**: Best for large models (SAM2 with batch size 4-8)
- **V100 (32GB)**: Adequate for smaller models
- **T4 (16GB)**: For inference only

### 2. Checkpoint Management

```bash
# Save checkpoints periodically
# In your config: checkpoint_interval: 10  # Every 10 epochs

# After training, copy important checkpoints to $WORK (backed up)
mkdir -p $WORK/vfm_checkpoints
cp finetune_logs/base_finetune-*/checkpoints/checkpoint_best.pt $WORK/vfm_checkpoints/
```

### 3. Efficient Data Transfer

**From Local to Grace:**
```bash
# Use rsync for resumable transfers
rsync -avzP data/ your_netid@grace.hprc.tamu.edu:$SCRATCH/vfm_project/data/

# For large files, use Globus (https://www.globus.org/)
```

**From Grace to Local:**
```bash
# Download results
rsync -avzP your_netid@grace.hprc.tamu.edu:$SCRATCH/vfm_project/results_segmentation/ ./results_local/
```

### 4. Logging Best Practices

```bash
# Use timestamps in logs
echo "Job started at: $(date)" >> logs/sam_finetune_$SLURM_JOB_ID.out

# Log environment info
echo "Python: $(which python)" >> logs/sam_finetune_$SLURM_JOB_ID.out
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')" >> logs/sam_finetune_$SLURM_JOB_ID.out
```

### 5. Interactive Sessions for Debugging

```bash
# Request interactive GPU session
srun --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash

# Once allocated, test your setup
module restore dl
source $SCRATCH/vfm_env/bin/activate
cd $SCRATCH/vfm_project
python -c "import torch; print(torch.cuda.is_available())"
```

### 6. Version Control

```bash
# Before submitting jobs, commit your changes
cd $SCRATCH/vfm_project
git add .
git commit -m "Training config for Grace HPC run"
git push

# Tag important runs
git tag -a v1.0-hpc-run -m "First successful HPC training run"
git push --tags
```

### 7. Reproducibility

```bash
# Log git commit hash in output
echo "Git commit: $(git rev-parse HEAD)" >> logs/sam_finetune_$SLURM_JOB_ID.out

# Save resolved Hydra config
# Hydra automatically saves to: finetune_logs/<experiment>/.hydra/config.yaml
```

---

## Quick Reference

### Essential Commands Cheat Sheet

```bash
# === VPN Connection (Off-Campus) ===
# Cisco AnyConnect: connect.tamu.edu
# Or: sudo openconnect connect.tamu.edu
ping grace.hprc.tamu.edu          # Verify VPN connection

# === SSH Connection ===
ssh your_netid@grace.hprc.tamu.edu

# === Module Management ===
module restore dl              # Load saved modules
module list                    # Show loaded modules
module avail python            # Search for Python modules

# === Environment ===
source $SCRATCH/vfm_env/bin/activate  # Activate venv
which python                          # Verify Python path

# === Navigation ===
cd $SCRATCH/vfm_project        # Go to project
cd $HOME                       # Go to home directory

# === Job Submission ===
sbatch scripts/slurm/run_training_grace.slurm        # Submit job
squeue -u $USER                        # Check queue
scancel 123456                         # Cancel job
tail -f logs/sam_finetune_123456.out   # Monitor output

# === Monitoring ===
watch -n 30 'squeue -u $USER'          # Watch queue
seff 123456                            # Job efficiency
scontrol show job 123456               # Job details

# === Storage ===
df -h $SCRATCH                 # Check disk usage
du -sh *                       # Size of directories
quota -s                       # Check quota

# === File Transfer ===
# From local to Grace:
scp file.txt your_netid@grace.hprc.tamu.edu:$SCRATCH/

# From Grace to local:
scp your_netid@grace.hprc.tamu.edu:$SCRATCH/results.json ./
```

---

## Complete Workflow Example

Here's a complete end-to-end workflow from scratch:

```bash
# ===== 1. INITIAL SETUP (First Time Only) =====

# Connect to TAMU VPN (if off-campus)
# Open Cisco AnyConnect and connect to connect.tamu.edu

# Connect to Grace
ssh shubhammhaske@grace.hprc.tamu.edu

# Create and save module collection
module purge
module load GCCcore/11.3.0 Python/3.10.4-GCCcore-11.3.0 CUDA/11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0
module save dl

# Create virtual environment
cd $SCRATCH
python3 -m venv vfm_env
source vfm_env/bin/activate
pip install --upgrade pip

# Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Clone and setup project
cd $SCRATCH
git clone https://github.com/shubham-mhaske/vfm_project.git
cd vfm_project
git submodule update --init --recursive
pip install -r requirements.txt

# Download checkpoints
bash sam2/checkpoints/download_ckpts.sh

# Setup data (assuming you have it locally)
mkdir -p data/bcss/{images,masks}
# Transfer data using scp or rsync from local machine

# Create log directories
mkdir -p logs finetune_logs results_segmentation

# ===== 2. SUBMIT JOB =====

# Verify everything is ready
ls -lh sam2/checkpoints/sam2.1_hiera_large.pt
ls data/bcss/images/*.png | wc -l  # Should be 151

# Submit training job
sbatch scripts/slurm/run_training_grace.slurm

# Note the job ID (e.g., 123456)

# ===== 3. MONITOR =====

# Check if job started
squeue -u $USER

# Monitor output
tail -f logs/sam_finetune_123456.out

# Check for errors
tail -f logs/sam_finetune_123456.err

# ===== 4. AFTER COMPLETION =====

# Check job efficiency
seff 123456

# Find best checkpoint
ls -lh finetune_logs/base_finetune-*/checkpoints/

# Run evaluation
sbatch scripts/slurm/evaluate_segmentation.slurm

# ===== 5. DOWNLOAD RESULTS =====

# From your local machine:
rsync -avzP shubhammhaske@grace.hprc.tamu.edu:$SCRATCH/vfm_project/results_segmentation/ ./results_local/
rsync -avzP shubhammhaske@grace.hprc.tamu.edu:$SCRATCH/vfm_project/finetune_logs/ ./finetune_logs_local/
```

---

## Additional Resources

- **HPRC Documentation**: https://hprc.tamu.edu/wiki/
- **Grace User Guide**: https://hprc.tamu.edu/wiki/Grace
- **SLURM Documentation**: https://slurm.schedmd.com/documentation.html
- **HPRC Help Portal**: https://hprc.tamu.edu/support/
- **Office Hours**: Check HPRC website for current schedule

## Getting Help

If you encounter issues:

1. **Check HPRC Status**: https://hprc.tamu.edu/status/
2. **Search HPRC Wiki**: https://hprc.tamu.edu/wiki/
3. **Submit Ticket**: helpdesk@hprc.tamu.edu
4. **Attend Office Hours**: Virtual and in-person available

---

**Document Version**: 1.0  
**Last Updated**: November 25, 2025  
**Maintained By**: VFM Project Team
