#!/bin/bash
#
# This script runs the full "Zero-Shot SAM" experiment matrix (1x3).
# It first generates the LLM prompt files and then runs the three
# evaluations, saving results to unique directories.
#
# It stops if any command fails.
set -e

# --- 1. API Key Check ---
# Check if the GEMINI_API_KEY is set.
if [ -z "$GEMINI_API_KEY" ]; then
    echo "================================================================================"
    echo "ERROR: GEMINI_API_KEY environment variable is not set."
    echo "Please set it before running this script:"
    echo "export GEMINI_API_KEY='YOUR_API_KEY_HERE'"
    echo "================================================================================"
    exit 1
fi
echo "✓ GEMINI_API_KEY is set."

# --- 2. Create Config Directories ---
# Ensure the configs/prompts and results directories exist.
mkdir -p configs/prompts
mkdir -p results
echo "✓ Directories are ready."

# --- 3. Move Hard-Coded Prompts (if necessary) ---
if [ -f "src/hard_coded_prompts.json" ]; then
    echo "Moving 'src/hard_coded_prompts.json' to 'configs/prompts/hard_coded_prompts.json'..."
    mv src/hard_coded_prompts.json configs/prompts/hard_coded_prompts.json
fi

# --- 4. Phase 1: Generate Prompt Configs ---
echo "================================================================================"
echo "Phase 1: Generating LLM Prompt Configurations..."
echo "================================================================================"

echo "\nRunning Text-Only LLM (gemini-pro-latest) to generate prompts..."
#python src/generate_text_prompts.py

echo "\nRunning Multimodal LLM (gemini-2.5-flash-preview-09-2025) to generate prompts..."
#python src/generate_multimodal_prompts.py

echo "\n✓ All prompt configuration files are ready."

# --- 5. Phase 2: Run "Zero-Shot SAM" Evaluations ---
echo "================================================================================"
echo "Phase 2: Running 'Zero-Shot SAM' Evaluation Matrix (1x3)..."
echo "================================================================================"

# --- Experiment 1 (Baseline) ---
echo "\nRunning Exp 1 (Zero-Shot SAM + Hard-Coded Prompts)..."
python src/evaluation.py \
    --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
    --clip_prompts configs/prompts/hard_coded_prompts.json \
    --output_dir results/exp1_zeroshot_sam_hardcoded

echo "✓ Exp 1 complete. Results saved to results/exp1_zeroshot_sam_hardcoded"

# --- Experiment 3 (Text-Only LLM) ---
echo "\nRunning Exp 3 (Zero-Shot SAM + Text-Only LLM Jargon)..."
python src/evaluation.py \
    --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
    --clip_prompts configs/prompts/llm_text_prompts_v1_gemini_pro_latest.json \
    --output_dir results/exp3_v1_zeroshot_sam_llm_text_jargon

echo "✓ Exp 3 complete. Results saved to results/exp3_v1_zeroshot_sam_llm_text_jargon"

# --- Experiment 5 (Multimodal LLM) ---
echo "\nRunning Exp 5 (Zero-Shot SAM + Multimodal LLM Jargon)..."
python src/evaluation.py \
    --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam_checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
    --clip_prompts configs/prompts/llm_multimodal_prompts_v1_gemini_2.5_flash.json \
    --output_dir results/exp5_v1_zeroshot_sam_llm_multimodal

echo "✓ Exp 5 complete. Results saved to results/exp5_v1_zeroshot_sam_llm_multimodal"

echo "================================================================================"
echo "All Zero-Shot experiments are complete."
echo "================================================================================"