**Repository Overview**
- **Purpose:** Medical image segmentation (BCSS) using SAM/SAM2. Focus is on reproducible evaluation, prompt-generation for CLIP/LLM, and finetuning SAM.
- **Top-level folders:** `src/` (primary code), `data/` (local datasets), `sam2/` (nested SAM2 repo & checkpoints), `CrowdsourcingDataset-Amgadetal2019/` (nested dataset repo), `models/` (checkpoints), `results/` (experiment outputs).

**How Code Runs (key patterns)**
- **Run from project root:** Most scripts expect the current working directory to be the project root (`vfm_project`) — imports and relative-path resolution assume this.
- **Path resolution:** Scripts (e.g., `src/evaluation.py`) resolve user args relative to `project_root` using `os.path.join(project_root, args.<arg>)`. When calling or editing scripts, prefer passing paths relative to repo root.
- **Argument pattern:** Top-level runners use `argparse` and then rewrite paths to absolute using the script's directory. Follow this pattern when adding new CLI flags.

**Major components & boundaries**
- **Prompt generation / LLM:** `src/generate_multimodal_prompts.py` (uses Gemini API, expects `GEMINI_API_KEY` in env); `src/prompt_generation.py` exists but is empty — treat `generate_multimodal_prompts.py` as the canonical multimodal prompt generator.
- **SAM integration:** `src/sam_segmentation.py` contains SAM/SAM2 predictor helpers; `sam2/` is a nested repo with model code and its own checkpoint downloader (`sam2/checkpoints/download_ckpts.sh`). For original ViT-H SAM checkpoint, check `models/sam_checkpoints/`.
- **Data loading:** `src/dataset.py` implements `BCSSDataset` and deterministic splits (special test-set selection by TCGA id prefixes — see README). Do not change the split logic without updating experiments.
- **Evaluation & experiments:** `src/evaluation.py` is the main quantitative runner (loads SAM2 predictor, CLIP classifier, iterates test set, writes `metrics.json` and `confusion_matrix.png`). Use `--output_dir` to capture results.

**Important files to inspect when changing behavior**
- `src/dataset.py` — data splits and loader output shape/keys (`image`, `mask`, `unique_classes`, `filename`, `image_np`).
- `src/sam_segmentation.py` — SAM predictor factory, prompt helpers, `get_predicted_mask_from_prompts`, `postprocess_mask`, `calculate_metrics`.
- `src/clip_classification.py` — CLIP-based classification utilities and `crop_region_from_mask` used in evaluation.
- `configs/` — prompt JSONs and model YAMLs (e.g., `configs/sam2.1/sam2.1_hiera_l.yaml`, `configs/prompts/*.json`). Use these canonical configs in experiments.

**Developer workflows / common commands**
- Install deps: `pip install -r requirements.txt` (run from repo root).
- Run full evaluation (example from repo context):
```
python src/evaluation.py \
  --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --clip_prompts configs/prompts/hard_coded_prompts_v2.json \
  --output_dir results/exp1_v2_zeroshot_sam_hardcoded_research
```
- Generate multimodal prompts (requires Gemini key):
```
export GEMINI_API_KEY="YOUR_KEY"
python src/generate_multimodal_prompts.py
```
- Download SAM2 checkpoints: `bash sam2/checkpoints/download_ckpts.sh` (run from repo root).

**Conventions & project-specific patterns**
- **Relative-to-root args:** CLI flags accept repo-relative paths; code resolves them using the script's location. Preserve this pattern for consistency.
- **No implicit imports:** Scripts frequently import sibling modules (e.g., `from dataset import BCSSDataset`). Ensure the project root is on `PYTHONPATH` or run from repo root.
- **Deterministic experiments:** Data splits use fixed rules/seed for reproducibility. Avoid ad-hoc shuffling when producing evaluation baselines.
- **Small utilities vs runners:** `src/` separates utility modules (`sam_segmentation.py`, `clip_classification.py`) from experiment runners (`evaluation.py`, `train_sam.py`). Edit utilities to change behavior; keep runners thin and focused on orchestration.

**LLM/Model integration gotchas**
- `generate_multimodal_prompts.py` uses `google.generativeai` and expects a model name compatible with multimodal inputs (the script contains a specific model string). If changing model name, test image+text prompts end-to-end.
- Environment variables: `GEMINI_API_KEY` is required for multimodal prompt generation. Other scripts assume local presence of large checkpoints and datasets — CI/containers must provision them.

**What to check before editing or running code**
- Confirm you have the nested repos (`sam2/`, `CrowdsourcingDataset-Amgadetal2019/`) checked out at the correct commits (README documents commit hashes).
- Ensure checkpoints exist in `sam2/checkpoints` and/or `models/sam_checkpoints`.
- For training: `tensorboard` is required but not always installed by default. If you see `ModuleNotFoundError: No module named 'tensorboard'`, run `pip install tensorboard`.
- Run small smoke runs on a subset of the dataset to validate changes (the dataset loader can be sliced for quick tests).

**If you need to modify experiments**
- Prefer changing helper functions in `src/sam_segmentation.py` and `src/clip_classification.py` rather than modifying runner plumbing.
- When adding CLI args, follow the pattern used in `src/evaluation.py`: add `argparse` flags, then canonicalize paths relative to `project_root` before use.

**Where to look for examples**
- Experiment results JSONs: `results/*/metrics.json` — useful for expected output schema.
- Notebooks: `notebooks/` — quick visual tests and examples of how functions are used during experimentation.

**Examples to copy/paste**
- Path-resolution snippet (use when adding CLI args):
```python
# resolve relative paths to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
args.some_path = os.path.join(project_root, args.some_path)
```

**When you are uncertain**
- Re-run the script from repo root. Many import/runtime issues are resolved by correct working directory.
- Inspect `README.md` for dataset/clone instructions and `requirements.txt` for required packages.

---
If you'd like, I can now: (1) run a quick smoke evaluation on 10 images, (2) add a short developer test harness, or (3) iterate on wording/details in this file. Which would you prefer?
