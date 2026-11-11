"""Entry point for Quoridor.

    python3 main.py          # graphical version (pygame)
    python3 main.py --cli    # terminal version
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Quoridor: Human vs. Human")
    parser.add_argument("--cli", action="store_true", help="play in the terminal instead of the GUI")
    parser.add_argument("--p1", default="Player 1", help="name of Player 1 (GUI)")
    parser.add_argument("--p2", default="Player 2", help="name of Player 2 (GUI)")
    args = parser.parse_args()

    if args.cli:
        from quoridor.cli import run_cli

        run_cli()
        return

    try:
        from quoridor.gui import run_gui
    except ImportError as exc:
        sys.exit(f"Could not start the GUI ({exc}).\nInstall it with: pip install -r requirements.txt\n"
                 "or play in the terminal with: python3 main.py --cli")
    run_gui(args.p1, args.p2)


if __name__ == "__main__":
    main()
