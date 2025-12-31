"""Command-line interface for NORA"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .core.config import load_config, merge_configs
from .ml.cli import add_ml_subparsers


def main(args: Optional[list] = None) -> int:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="NORA: Norm-Aware Agent for Reliable ML Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    # Create subparsers for different command groups
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ML pipeline commands
    ml_subparsers = subparsers.add_parser("ml", help="ML pipeline commands").add_subparsers(
        dest="ml_command"
    )
    add_ml_subparsers(ml_subparsers)

    # NORA training commands
    train_parser = subparsers.add_parser(
        "train",
        help="Train NORA with neural networks (vision models)",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base config file",
    )
    train_parser.add_argument(
        "--regime",
        type=str,
        choices=["strict", "balanced", "exploratory"],
        help="Norm enforcement regime",
    )
    train_parser.add_argument(
        "--mode",
        type=str,
        choices=["pipeline", "agentic", "norm_aware"],
        help="Execution mode",
    )
    train_parser.add_argument(
        "--violation",
        type=str,
        choices=[
            "none",
            "nondeterminism",
            "amp_nan",
            "eval_mode_bug",
            "aug_leak",
            "checkpoint_incomplete",
        ],
        help="Violation to inject (if any)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of training epochs",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Output directory for runs",
    )
    train_parser.set_defaults(func=cmd_train)

    parsed_args = parser.parse_args(args)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, parsed_args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Route to appropriate command
    if parsed_args.command == "ml":
        if hasattr(parsed_args, "func"):
            return parsed_args.func(parsed_args)
        else:
            print("Error: ML subcommand required. Use 'nora ml --help'")
            return 1
    elif parsed_args.command == "train":
        return cmd_train(parsed_args)
    else:
        parser.print_help()
        return 0


def cmd_train(args) -> int:
    """Execute NORA training command"""
    logger = logging.getLogger(__name__)
    logger.info("Loading NORA configuration")

    # Load configs
    config = load_config(args.config)

    # Merge regime if specified
    if args.regime:
        regime_path = f"configs/regimes/{args.regime}.yaml"
        regime_config = load_config(regime_path)
        config = merge_configs(config, regime_config)

    # Merge mode if specified
    if args.mode:
        mode_path = f"configs/modes/{args.mode}.yaml"
        mode_config = load_config(mode_path)
        config = merge_configs(config, mode_config)

    # Merge violation if specified
    if args.violation:
        violation_path = f"configs/violations/{args.violation}.yaml"
        violation_config = load_config(violation_path)
        config = merge_configs(config, violation_config)

    # Override with CLI args
    if args.seed:
        config["reproducibility"]["seed"] = args.seed
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size

    logger.info(f"Configuration loaded")
    logger.info("NORA training initialized (framework is under construction)")
    return 0


if __name__ == "__main__":
    main()
