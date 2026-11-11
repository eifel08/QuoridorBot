"""Entry point for Quoridor.

    python3 main.py          # graphical version (pygame)
    python3 main.py --cli    # terminal version
    python3 main.py --web    # browser version on http://localhost:8000
"""

import argparse
import sys


def serve_web(port: int, open_browser: bool = True) -> None:
    """Serve the static web/ folder (no dependencies beyond the stdlib)."""
    import functools
    import http.server
    import webbrowser
    from pathlib import Path

    web_dir = Path(__file__).resolve().parent / "web"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web_dir))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}/"
        print(f"Quoridor is running at {url}  (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Quoridor: Human vs Human or Human vs Bot")
    parser.add_argument("--cli", action="store_true", help="play in the terminal instead of the GUI")
    parser.add_argument("--web", action="store_true", help="serve the browser version locally")
    parser.add_argument("--port", type=int, default=8000, help="port for --web (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="with --web, don't open a browser tab")
    parser.add_argument("--mode", choices=["human", "bot"],
                        help="skip the menu: 'human' = Human vs Human, 'bot' = Human vs Bot")
    parser.add_argument("--depth", type=int, default=4, help="bot search depth 1-30 (default 4)")
    parser.add_argument("--time-limit", type=float, default=5.0,
                        help="bot think time per move in seconds, 0 = no limit (default 5)")
    parser.add_argument("--play-as", type=int, choices=[1, 2], default=1,
                        help="against the bot: play as Player 1 (moves first) or 2")
    parser.add_argument("--ml-model", help="trained checkpoint (.pt): the bot is the ML agent (needs PyTorch)")
    parser.add_argument("--ml-sims", type=int, default=64, help="MCTS simulations per ML-agent move")
    parser.add_argument("--p1", default="Player 1", help="name of Player 1 (GUI)")
    parser.add_argument("--p2", default="Player 2", help="name of Player 2 (GUI)")
    args = parser.parse_args()
    if not 1 <= args.depth <= 30:
        parser.error("--depth must be between 1 and 30")
    mode = {"human": "hvh", "bot": "hvb", None: None}[args.mode]
    time_limit = args.time_limit or None
    if args.ml_model and mode is None:
        mode = "hvb"

    if args.web:
        serve_web(args.port, open_browser=not args.no_browser)
        return

    if args.cli:
        from quoridor.cli import run_cli

        run_cli(mode, args.depth, time_limit, args.play_as - 1, args.ml_model, args.ml_sims)
        return

    try:
        from quoridor.gui import run_gui
    except ImportError as exc:
        sys.exit(f"Could not start the GUI ({exc}).\nInstall it with: pip install -r requirements.txt\n"
                 "or play in the terminal with: python3 main.py --cli")
    from quoridor.gui import MatchSettings

    settings = MatchSettings(mode or "hvh", args.depth, time_limit, args.play_as - 1, args.ml_model, args.ml_sims)
    run_gui(args.p1, args.p2, settings=settings, show_menu=mode is None)


if __name__ == "__main__":
    main()
