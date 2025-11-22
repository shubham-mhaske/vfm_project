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

    # --- Optional Resume Logic ---
    resume_path = cfg.get('resume_checkpoint', None)
    if resume_path and os.path.isfile(resume_path):
        print(f"[Resume] Loading checkpoint: {resume_path}")
        import torch
        try:
            ckpt = torch.load(resume_path, map_location='cpu')
            # Model state
            model_state = ckpt.get('model') or ckpt.get('model_state_dict')
            if model_state:
                missing, unexpected = trainer.model.load_state_dict(model_state, strict=False)
                print(f"[Resume] Model state loaded (missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                print("[Resume] No model state found in checkpoint.")
            # Optimizer state
            if hasattr(trainer, 'optimizer') and 'optimizer' in ckpt:
                try:
                    trainer.optimizer.load_state_dict(ckpt['optimizer'])
                    print("[Resume] Optimizer state loaded.")
                except Exception as e:
                    print(f"[Resume] Failed to load optimizer state: {e}")
            # LR scheduler state(s)
            if hasattr(trainer, 'lr_schedulers') and 'lr_schedulers' in ckpt:
                try:
                    # Expect list/dict of schedulers
                    sched_state = ckpt['lr_schedulers']
                    if isinstance(sched_state, (list, tuple)) and isinstance(trainer.lr_schedulers, (list, tuple)):
                        for sch, state in zip(trainer.lr_schedulers, sched_state):
                            sch.load_state_dict(state)
                    elif isinstance(sched_state, dict):
                        # Single scheduler
                        trainer.lr_schedulers.load_state_dict(sched_state)
                    print("[Resume] LR scheduler state loaded.")
                except Exception as e:
                    print(f"[Resume] Failed to load LR scheduler state: {e}")
            # Step / epoch bookkeeping (if present)
            trainer_start_epoch = ckpt.get('epoch')
            if trainer_start_epoch is not None and hasattr(trainer, 'start_epoch'):
                trainer.start_epoch = int(trainer_start_epoch) + 1
                print(f"[Resume] Resuming from epoch {trainer_start_epoch}, next epoch {trainer.start_epoch}.")
        except Exception as e:
            print(f"[Resume] Error loading checkpoint: {e}")
    elif resume_path:
        print(f"[Resume] Provided resume_checkpoint path not found: {resume_path}")

    # --- Run Training ---
    print("Starting training...")
    trainer.run()
    print("Training finished successfully.")
    print(f"Output logs and checkpoints saved in: {os.getcwd()}")


if __name__ == "__main__":
    main()
