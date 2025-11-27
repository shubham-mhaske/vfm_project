#!/usr/bin/env python3
"""
Finds the best checkpoint from a training run directory based on per-class
validation results.
"""

import os
import json
import argparse
import numpy as np

def find_best_checkpoint(run_dir):
    """
    Parses validation JSON files in a run directory to find the checkpoint
    with the best overall Dice score.

    Args:
        run_dir (str): The main directory of a training run, containing
                       a 'validation' subdirectory with JSON results.

    Returns:
        str: The absolute path to the best checkpoint file, or None.
    """
    validation_dir = os.path.join(run_dir, 'validation_results')
    if not os.path.isdir(validation_dir):
        print(f"Error: Validation directory not found at '{validation_dir}'")
        return None

    best_score = -1.0
    best_checkpoint_path = None

    print(f"Scanning for validation results in: {validation_dir}")

    for filename in os.listdir(validation_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(validation_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                overall_dice = data.get('overall')
                if overall_dice is not None and overall_dice > best_score:
                    best_score = overall_dice
                    # Filename is expected to be like 'checkpoint_10.json'
                    ckpt_name = os.path.splitext(filename)[0] + '.pt'
                    # The actual checkpoint is in the 'checkpoints' dir
                    ckpt_path = os.path.join(run_dir, 'checkpoints', ckpt_name)
                    if os.path.exists(ckpt_path):
                        best_checkpoint_path = ckpt_path

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {filepath}: {e}")

    if best_checkpoint_path:
        print(f"\nBest Checkpoint Found:")
        print(f"  Path: {best_checkpoint_path}")
        print(f"  Overall Dice: {best_score:.4f}")
    else:
        print("\nNo valid checkpoints or validation results found.")

    return best_checkpoint_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the best checkpoint from a training run.')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Directory of the training run (e.g., finetune_logs/my_run-timestamp).')
    
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    run_dir_abs = os.path.join(project_root, args.run_dir)
    
    best_checkpoint = find_best_checkpoint(run_dir_abs)
    
    # The primary output of this script is the path to the best checkpoint,
    # so we print it to stdout for use in shell scripts.
    if best_checkpoint:
        print(f"\n{best_checkpoint}")
