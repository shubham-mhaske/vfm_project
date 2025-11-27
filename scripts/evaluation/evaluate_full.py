#!/usr/bin/env python3
"""
Runs a complete evaluation pipeline for a given training run.
This script orchestrates the following steps:
1. Finds the best checkpoint from validation scores.
2. Determines optimal per-class thresholds for that checkpoint.
3. Runs a final evaluation on the test set using TTA and the optimal thresholds.
4. Generates visualizations of predictions.
5. Updates the main EXPERIMENTS.md file with the results.
"""

import os
import sys
import json
import argparse
import subprocess
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

def run_command(command, cwd=project_root):
    """Executes a shell command and returns its output."""
    print(f"Running command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if process.returncode != 0:
        print(f"Error running command: {' '.join(command)}")
        print(process.stderr)
        raise RuntimeError(f"Command failed with exit code {process.returncode}")
    return process.stdout.strip()

def update_experiments_md(run_name, metrics_file):
    """Appends the results from a metrics file to EXPERIMENTS.md."""
    if not os.path.exists(metrics_file):
        print(f"Warning: Metrics file not found at {metrics_file}. Skipping update of EXPERIMENTS.md.")
        return

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Format results into a markdown table row
    # This is a simplified example. A real implementation would need to parse
    # per-class metrics if they are available in the JSON.
    overall_dice = metrics.get('avg_dice', 0.0)
    
    # Create a new section if it doesn't exist
    md_content = f"""
### {run_name}
**Date**: {time.strftime("%Y-%m-%d")}
**Checkpoint**: Best from run
**Metrics**:
| Class   | Dice  |
|---------|-------|
| Overall | {overall_dice:.4f} |
"""
    # In a real scenario, you'd parse per-class results here.

    with open(os.path.join(project_root, 'EXPERIMENTS.md'), 'a') as f:
        f.write(md_content)
    
    print("Updated EXPERIMENTS.md")

def main(args):
    """Main orchestration function."""
    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(project_root, run_dir)

    if not os.path.isdir(run_dir):
        print(f"Error: Run directory not found at '{run_dir}'")
        sys.exit(1)

    run_name = os.path.basename(run_dir)
    print(f"Starting full evaluation for run: {run_name}")

    # 1. Find best checkpoint
    print("\n--- Step 1: Finding best checkpoint ---")
    best_ckpt_script = os.path.join(project_root, 'scripts/validation/find_best_checkpoint.py')
    best_ckpt_path = run_command(['python', best_ckpt_script, '--run_dir', run_dir]).splitlines()[-1]
    
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        print("Error: Could not determine the best checkpoint.")
        sys.exit(1)
    
    # 2. Find optimal thresholds
    print("\n--- Step 2: Finding optimal thresholds ---")
    threshold_script = os.path.join(project_root, 'scripts/validation/find_optimal_thresholds.py')
    threshold_config = os.path.join(project_root, 'configs', f'optimal_thresholds_{run_name}.json')
    run_command(['python', threshold_script, '--checkpoint', best_ckpt_path, '--output', threshold_config])

    # 3. Run final evaluation
    print("\n--- Step 3: Running final evaluation ---")
    eval_script = os.path.join(project_root, 'src/evaluate_segmentation.py')
    eval_output_dir = os.path.join(project_root, 'results', f'{run_name}_final_eval')
    
    eval_command = [
        'python', eval_script,
        '--checkpoint', best_ckpt_path,
        '--split', 'test',
        '--use_tta',
        '--threshold_config', threshold_config,
        '--output_dir', eval_output_dir,
        '--save_predictions' # New argument to save visualizations
    ]
    run_command(eval_command)
    
    # 4. Update EXPERIMENTS.md
    print("\n--- Step 4: Updating documentation ---")
    metrics_file = os.path.join(eval_output_dir, 'metrics.json')
    update_experiments_md(run_name, metrics_file)

    print("\n✅ Full evaluation pipeline complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the complete evaluation pipeline.')
    parser.add_argument('run_dir', type=str, help='Directory of the training run.')
    args = parser.parse_args()
    main(args)
