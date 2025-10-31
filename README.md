# Visual Foundation Model (VFM) Project

This project is designed for medical image segmentation tasks, specifically focusing on breast cancer histology images. It utilizes the Segment Anything Model (SAM/SAM2) and provides a framework for finetuning, evaluation, and running baseline comparisons.

## Code Structure

The project is organized into the following key directories:

-   `/src`: Contains all the primary Python source code for the project.
    -   `dataset.py`: Defines PyTorch `Dataset` classes for loading and preprocessing data.
    -   `prompt_generation.py`: Intended for generating prompts (e.g., points, boxes) for the SAM model from masks. **(Note: This file is currently empty).**
    -   `sam_segmentation.py`: Contains the core logic for performing segmentation using the SAM model.
    -   `train_sam.py`: The main script for finetuning the SAM model on a custom dataset.
    -   `evaluate_segmentation.py`: The main script for evaluating the performance of a trained segmentation model.
-   `/data`: The default directory where all datasets are stored.
    -   `bcss/`: Intended for the Breast Cancer Semantic Segmentation (BCSS) dataset.
-   `/sam2`: A Git submodule containing the SAM2 model source code and checkpoints.
-   `/CrowdsourcingDataset-Amgadetal2019`: A Git submodule containing scripts to download and manage the crowdsourced breast cancer dataset from Amgad et al., 2019.
-   `/notebooks`: Jupyter notebooks for experimentation, visualization, and model testing.
-   `/models`: Directory for storing model checkpoints, both pre-trained and finetuned.

## Dataset Setup

The project uses the Breast Cancer Semantic Segmentation (BCSS) dataset, which was generated via crowdsourcing by Amgad et al., 2019. The `BCSSDataset` class in `src/dataset.py` is configured to load this data from the `/data/bcss` directory.

Follow these steps to download and set up the dataset:

### Step 1: Download the Data

The `CrowdsourcingDataset-Amgadetal2019` submodule provides a script to download the images and masks.

1.  Navigate to the submodule directory:
    ```bash
    cd CrowdsourcingDataset-Amgadetal2019
    ```
2.  Install the required Python packages:
    ```bash
    pip install girder_client numpy imageio scikit-image
    ```
3.  Run the download script:
    ```bash
    python download_crowdsource_dataset.py
    ```
4.  The script will prompt you for authentication to the Girder API. You can register for a free account if you do not have one.

This will download the data into the `CrowdsourcingDataset-Amgadetal2019/data` directory.

### Step 2: Organize the Data for the Project

The `BCSSDataset` loader expects the files in the root `/data/bcss` directory. You will need to move or copy the downloaded files to the correct location.

1.  Move the downloaded images to `/data/bcss/images/`.
2.  Move the downloaded masks to `/data/bcss/masks/`.

The `BCSSDataset` class will then automatically handle the splitting of data into training, validation, and test sets.

## About the BCSS Dataset

The Breast Cancer Semantic Segmentation (BCSS) dataset from Amgad et al., 2019, consists of high-resolution image patches extracted from breast cancer histology slides. It is designed for semantic segmentation tasks.

### File Naming Convention

Each image and mask filename provides important metadata:
`TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png`

-   `TCGA-A1-A0SK-DX1`: The identifier for the original whole-slide image from The Cancer Genome Atlas (TCGA).
-   `xmin..._ymin...`: The x and y coordinates of the top-left corner of the patch within the original slide.
-   `MPP-0.2500`: The microns-per-pixel resolution at which the patch was saved.

### Image and Mask Format

-   **Images**: Standard RGB PNG files.
-   **Masks**: Single-channel PNG files where each pixel's integer value represents a specific tissue class.

### Class Definitions

The pixel values in the masks correspond to the following classes:

-   `0`: Background
-   `1`: Tumor
-   `2`: Stroma
-   `3`: Lymphocyte
-   `4`: Necrosis
-   `5`: Blood Vessel

## Data Processing

1.  **Data Loading & Splitting:** The `src/dataset.py` file defines the `BCSSDataset` class, which is responsible for loading and preparing the data. It performs a deterministic split to ensure reproducible experiments:
    -   **Test Set**: A fixed test set is created using all images whose TCGA identifiers start with `TCGA-OL-`, `TCGA-LL-`, `TCGA-E2-`, `TCGA-EW-`, `TCGA-GM-`, or `TCGA-S3-`. This is critical for ensuring all baseline comparisons are evaluated on the same unseen data.
    -   **Training & Validation Sets**: All remaining files are shuffled reproducibly (using a fixed random seed), with 80% used for training and 20% for validation.

2.  **Data Structure:** When you iterate over a `BCSSDataset` object (e.g., in a PyTorch `DataLoader`), it returns a dictionary for each image-mask pair with the following structure:

    ```python
    {
        'image': torch.Tensor,      # The transformed image tensor (C, H, W)
        'mask': torch.Tensor,       # The mask tensor (H, W)
        'unique_classes': np.array, # A NumPy array of class IDs present in this mask
        'filename': str,            # The original filename of the image
        'image_np': np.array        # The original image as a NumPy array (H, W, C)
    }
    ```


## New User Setup and Repository Guide

This guide explains the repository's structure and provides a step-by-step process for setting up the project environment.

### Repository vs. Local Files

It is important to distinguish between files tracked by Git and files that are generated locally.

**What is in the Repository (Tracked by Git):**
-   `/src`, `/notebooks`, `/configs`: All source code, experiment notebooks, and configuration files.
-   `requirements.txt`: A list of Python packages required for the project.
-   `/sam2` & `/CrowdsourcingDataset-Amgadetal2019`: These are Git submodules. The repository only stores a pointer to a specific commit of these external projects, not all their files.

**What is NOT in the Repository (Local Files):**
-   `/data`: Contains the large BCSS dataset. This is downloaded locally.
-   `/models` & `/checkpoints`: Contain large, pre-trained model weights. These are downloaded locally.
-   `/finetune_logs`: Contains outputs from training runs (logs, TensorBoard files). These are generated during execution.
-   Python cache (`__pycache__`), build artifacts (`*.egg-info`), and local environment folders (`.venv`).

### Step-by-Step Setup Instructions

**A Note on Submodules:** This project uses Git submodules to include the `sam2` and `CrowdsourcingDataset-Amgadetal2019` repositories. A submodule locks the external code to a specific version (commit). This is a deliberate choice to ensure **reproducibility**—it guarantees that every team member is using the exact same version of the dependency code, which is critical for consistent baseline results. The alternative would be a script that clones the repositories, which could accidentally download newer, incompatible versions.

**1. Clone the Repository**

Clone the repository and initialize the submodules (`sam2` and `CrowdsourcingDataset-Amgadetal2019`) in one command:

```bash
git clone --recurse-submodules <your-repository-url>
```

*If you have already cloned the repository without the flag, you can initialize the submodules with:*
```bash
git submodule update --init --recursive
```

**2. Install Dependencies**

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**3. Download Pre-trained Models**

The project uses checkpoints from both SAM and SAM2.

-   **SAM2 Models:** The `sam2` submodule includes a script to download its models. Run it from the project root:
    ```bash
    bash sam2/checkpoints/download_ckpts.sh
    ```
-   **Original SAM Model:** The project also uses the original ViT-H SAM model. Create the directory for it and download the checkpoint from the official source.
    ```bash
    mkdir -p models/sam_checkpoints
    # Download sam_vit_h_4b8939.pth and place it in models/sam_checkpoints/
    # The model can be found in the official Segment Anything repository.
    ```

**4. Download the Dataset**

Follow the instructions in the **Dataset Setup** section above to download and organize the BCSS dataset in the `/data/bcss` directory.

After completing these steps, your local environment will be fully set up and ready for running the training and evaluation scripts.
