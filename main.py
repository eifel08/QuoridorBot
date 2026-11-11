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
    parser = argparse.ArgumentParser(description="Quoridor: Human vs. Human")
    parser.add_argument("--cli", action="store_true", help="play in the terminal instead of the GUI")
    parser.add_argument("--web", action="store_true", help="serve the browser version locally")
    parser.add_argument("--port", type=int, default=8000, help="port for --web (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="with --web, don't open a browser tab")
    parser.add_argument("--p1", default="Player 1", help="name of Player 1 (GUI)")
    parser.add_argument("--p2", default="Player 2", help="name of Player 2 (GUI)")
    args = parser.parse_args()

    if args.web:
        serve_web(args.port, open_browser=not args.no_browser)
        return

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
