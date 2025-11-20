import os
import sys
import argparse
import subprocess

def main(args, passthrough):
    """Launches the SAM 2 fine-tuning job with the specified configuration."""
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sam2_root = os.path.join(project_root, 'sam2')
    
    # Prepend sam2/ and project root to PYTHONPATH
    # This allows Python to find:
    # - sam2.sam2.configs module (needed by Hydra)
    # - training.utils module (needed by train.py)
    # - src.finetune_dataset module (needed by config)
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{sam2_root}:{project_root}:{env.get('PYTHONPATH', '')}"
    env['HYDRA_FULL_ERROR'] = '1'
    
    train_script_path = os.path.join(project_root, args.train_script)
    
    command = [
        "python",
        train_script_path,
        "-c",
        args.config_name,
        *passthrough,
    ]
    
    print(f"Executing command: {' '.join(command)}")
    print("This will start the fine-tuning process. It may take a long time and requires a GPU.")
    print("Monitor the output for training progress and potential errors.")
    
    # Execute the command
    subprocess.run(command, env=env, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch SAM 2 fine-tuning job.")
    parser.add_argument('-c', '--config_name', type=str, default="bcss_finetune", help='Name of the configuration to use.')
    parser.add_argument('--train_script', type=str, default="sam2/training/train.py", help='Path to the training script.')
    # Capture any additional Hydra/override args to forward to the train script
    args, passthrough = parser.parse_known_args()
    main(args, passthrough)