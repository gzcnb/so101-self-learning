"""Start the desktop console and read the SSH password interactively when needed."""

import getpass
import os
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workbench"))
from settings import BOARD_HOST


def main():
    if BOARD_HOST and "SO101_SSH_PASSWORD" not in os.environ:
        os.environ["SO101_SSH_PASSWORD"] = getpass.getpass("Orange Pi SSH password: ")
    from swap_console import main as serve
    serve()


if __name__ == "__main__":
    main()
