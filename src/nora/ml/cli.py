"""ML pipeline CLI commands"""

import argparse
import logging
from pathlib import Path

from ..ml.validate_data import validate_data_file
from ..ml.build_features import build_features
from ..ml.train import train
from ..ml.evaluate import evaluate

logger = logging.getLogger(__name__)


def add_ml_subparsers(subparsers):
    """Add ML pipeline subcommands"""

    # Validate data command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate data against schema",
    )
    validate_parser.add_argument("--input", required=True, help="Input CSV file")
    validate_parser.add_argument(
        "--schema",
        required=True,
        help="Schema YAML file",
    )
    validate_parser.set_defaults(func=cmd_validate)

    # Build features command
    features_parser = subparsers.add_parser(
        "build-features",
        help="Build features from raw data",
    )
    features_parser.add_argument("--input", required=True, help="Input CSV file")
    features_parser.add_argument(
        "--output",
        required=True,
        help="Output directory for parquet files",
    )
    features_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    features_parser.set_defaults(func=cmd_build_features)

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train ML model",
    )
    train_parser.add_argument(
        "--input",
        required=True,
        help="Input parquet file (training data)",
    )
    train_parser.add_argument(
        "--config",
        required=True,
        help="Model config YAML",
    )
    train_parser.add_argument(
        "--output",
        required=True,
        help="Output model path",
    )
    train_parser.set_defaults(func=cmd_train)

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model on test set",
    )
    eval_parser.add_argument(
        "--model",
        required=True,
        help="Trained model path",
    )
    eval_parser.add_argument(
        "--test-set",
        required=True,
        help="Test set parquet file",
    )
    eval_parser.add_argument(
        "--output",
        required=True,
        help="Output metrics JSON",
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    return subparsers


def cmd_validate(args):
    """Execute validate command"""
    logger.info("Validating data...")
    valid = validate_data_file(args.input, args.schema)
    return 0 if valid else 1


def cmd_build_features(args):
    """Execute build-features command"""
    logger.info("Building features...")
    build_features(args.input, args.output, args.seed)
    return 0


def cmd_train(args):
    """Execute train command"""
    logger.info("Training model...")
    train(args.input, args.config, args.output)
    return 0


def cmd_evaluate(args):
    """Execute evaluate command"""
    logger.info("Evaluating model...")
    evaluate(args.model, args.test_set, args.output)
    return 0
