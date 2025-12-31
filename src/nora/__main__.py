"""NORA command-line entry point"""

from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main() or 0)
