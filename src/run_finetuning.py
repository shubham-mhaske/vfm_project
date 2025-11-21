import os
import sys
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging

# Add project root and sam2 to python path to allow importing modules from them.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sam2_root = os.path.join(project_root, 'sam2')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if sam2_root not in sys.path:
    sys.path.insert(0, sam2_root)

# The original train script registers custom OmegaConf resolvers (e.g., for 'times').
# We need to do the same so that the configs can be resolved correctly.
try:
    from training.utils.train_utils import register_omegaconf_resolvers
except ImportError as e:
    print(f"Could not import 'register_omegaconf_resolvers'. Make sure 'sam2' is in the PYTHONPATH.")
    print(f"PYTHONPATH: {sys.path}")
    raise e

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the fine-tuning process using the structured Hydra configuration.

    To run an experiment, use the command line, for example:
    `python src/run_finetuning.py experiment=low_lr`
    `python src/run_finetuning.py experiment=strong_aug scratch.train_batch_size=4`
    """
    # --- Setup ---
    # Register custom OmegaConf resolvers (e.g., for the 'times' resolver).
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.warning(f"Could not register OmegaConf resolvers: {e}")

    print("--- Experiment Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------------")

    # The original script handles multi-gpu, but we focus on a single process run
    # which is simpler and covers the single-GPU case.
    # These environment variables are set even for a single GPU.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)  # A default port
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # --- Instantiate Trainer ---
    # The config is already composed by Hydra's @main decorator.
    # We just need to instantiate the trainer object from the config.
    print("Instantiating trainer...")
    trainer = instantiate(cfg.trainer, _recursive_=False)

    # --- Run Training ---
    print("Starting training...")
    trainer.run()
    print("Training finished successfully.")
    print(f"Output logs and checkpoints saved in: {os.getcwd()}")


if __name__ == "__main__":
    main()
