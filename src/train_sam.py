import os
import sys

def main():
    """Launches the SAM 2 fine-tuning job with the custom BCSS configuration."""
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Prepend the project root to the PYTHONPATH environment variable
    python_path = os.environ.get('PYTHONPATH', '')
    python_path = f"{project_root}:{python_path}"
    
    config_name = "bcss_finetune"
    train_script_path = "sam2/training/train.py"
    
    command = f"PYTHONPATH={python_path} python {train_script_path} -c {config_name}"
    
    print(f"Executing command: {command}")
    print("This will start the fine-tuning process. It may take a long time and requires a GPU.")
    print("Monitor the output for training progress and potential errors.")
    
    # Execute the command
    os.system(command)

if __name__ == '__main__':
    main()