"""Command-line interface for NORA"""

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from .ml.cli import add_ml_subparsers
from .train.runner import run_training

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training entry point with Hydra"""
    logger.info("Starting NORA training")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Convert OmegaConf to dict for compatibility
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Determine output directory from Hydra's working directory
    output_dir = str(Path.cwd())
    
    # Run training
    result = run_training(config, output_dir=output_dir)
    logger.info(
        f"Training finished: best_epoch={result.get('best_epoch')}, "
        f"best_val_acc={result.get('best_val_acc'):.4f}"
    )


def main_ml() -> int:
    """ML pipeline CLI (non-Hydra, legacy support)"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NORA ML Pipeline")
    subparsers = parser.add_subparsers(dest="ml_command", help="ML command to run")
    add_ml_subparsers(subparsers)
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 1


def main(args: Optional[list] = None) -> int:
    """Main CLI entry point - routes to appropriate command"""
    # Simple command detection before full parsing
    if args is None:
        args = sys.argv[1:]
    
    if len(args) > 0 and args[0] == "ml":
        # ML pipeline - use legacy argparse
        sys.argv = [sys.argv[0]] + args[1:]
        return main_ml()
    elif len(args) > 0 and args[0] == "train":
        # Training - use Hydra
        sys.argv = [sys.argv[0]] + args[1:]
        train()
        return 0
    else:
        # No command - show help
        print("NORA: Norm-Aware Agent for Reliable ML Training")
        print("\nCommands:")
        print("  train  - Train vision models with norm awareness")
        print("  ml     - Run ML pipeline (tabular data)")
        print("\nExamples:")
        print("  python -m nora train mode=norm_aware regime=balanced")
        print("  python -m nora ml validate --input data.csv --schema schema.yaml")
        return 0


if __name__ == "__main__":
    sys.exit(main())
