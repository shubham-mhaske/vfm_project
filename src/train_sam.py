import os
import sys
import argparse
import subprocess

def main(args):
    """Launches the SAM 2 fine-tuning job with the specified configuration."""
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Prepend the project root to the PYTHONPATH environment variable
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    
    train_script_path = os.path.join(project_root, args.train_script)
    
    command = [
        "python",
        train_script_path,
        "-c",
        args.config_name
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
    args = parser.parse_args()
    main(args)