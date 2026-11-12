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
    parser.add_argument("--mode", choices=["human", "bot", "bots"],
                        help="skip the menu: 'human' = Human vs Human, 'bot' = Human vs Bot, "
                             "'bots' = Bot vs Bot")
    parser.add_argument("--depth", type=int, default=4, help="bot search depth 1-30 (default 4)")
    parser.add_argument("--time-limit", type=float, default=5.0,
                        help="bot think time per move in seconds, 0 = no limit (default 5)")
    parser.add_argument("--engine", choices=["auto", "cpp", "py"], default=None,
                        help="bot search engine: C++ (engine/, must be built), pure Python, or auto = "
                             "C++ if built (default)")
    parser.add_argument("--walls", choices=["attack", "both", "all"], default=None,
                        help="walls the bot's search considers: attack = only walls that lengthen the "
                             "opponent's shortest paths, both = attack + defense walls (default), all = "
                             "every legal wall. Applies to the Human vs Bot bot and to --bot1/--bot2 specs "
                             "that don't set walls= themselves (C++ engine only)")
    parser.add_argument("--play-as", type=int, choices=[1, 2], default=1,
                        help="against the bot: play as Player 1 (moves first) or 2")
    parser.add_argument("--ml-model", help="trained checkpoint (.pt): the bot is the ML agent (needs PyTorch)")
    parser.add_argument("--ml-sims", type=int, default=64, help="MCTS simulations per ML-agent move")
    parser.add_argument("--bot1", help="Bot vs Bot: spec of Bot 1 / Red, e.g. ab:depth=2 (default)")
    parser.add_argument("--bot2", help="Bot vs Bot: spec of Bot 2 / Blue, e.g. ab:depth=4,time=2 (default "
                                       "ab:depth=4) or ml:model=best.pt")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Bot vs Bot (watching): pause between moves in seconds (default 0.5)")
    parser.add_argument("--headless", action="store_true",
                        help="Bot vs Bot: no board, play many games fast and print statistics; extra "
                             "options go to quoridor.botmatch (--games, --seed, --json, ...)")
    parser.add_argument("--p1", default="Player 1", help="name of Player 1 (GUI)")
    parser.add_argument("--p2", default="Player 2", help="name of Player 2 (GUI)")
    args, extra = parser.parse_known_args()
    if extra and not args.headless:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if not 1 <= args.depth <= 30:
        parser.error("--depth must be between 1 and 30")
    if args.engine:
        from quoridor.bots import cpp_engine

        cpp_engine.default_engine = args.engine
        if args.engine == "cpp" and not cpp_engine.engine_available():
            parser.error("the C++ engine is not built; run: python build_engine.py")
    mode = {"human": "hvh", "bot": "hvb", "bots": "bvb", None: None}[args.mode]
    time_limit = args.time_limit or None
    if args.ml_model and mode is None:
        mode = "hvb"
    if args.headless:
        from quoridor.botmatch import main as botmatch

        botmatch(["--bot1", args.bot1 or "ab:depth=2", "--bot2", args.bot2 or "ab:depth=4", *extra])
        return
    from quoridor.bots import BotSpec, parse_bot_spec

    try:
        specs = (parse_bot_spec(args.bot1) if args.bot1 else BotSpec(depth=2),
                 parse_bot_spec(args.bot2) if args.bot2 else BotSpec(depth=4))
    except ValueError as exc:
        parser.error(str(exc))
    wall_filter = None
    if args.walls:
        from quoridor.bots.cpp_engine import wall_filter_from_name

        wall_filter = wall_filter_from_name(args.walls)
        specs = tuple(sp.with_wall_filter(wall_filter)
                      if sp.kind == "ab" and "wall_filter" not in dict(sp.options) else sp for sp in specs)

    if args.web:
        serve_web(args.port, open_browser=not args.no_browser)
        return

    if args.cli:
        from quoridor.cli import run_cli

        run_cli(mode, args.depth, time_limit, args.play_as - 1, args.ml_model, args.ml_sims,
                specs, args.delay, wall_filter)
        return

    try:
        from quoridor.gui import run_gui
    except ImportError as exc:
        sys.exit(f"Could not start the GUI ({exc}).\nInstall it with: pip install -r requirements.txt\n"
                 "or play in the terminal with: python3 main.py --cli")
    from quoridor.gui import MatchSettings

    menu_mode = mode or ("bvb" if args.bot1 or args.bot2 else "hvh")
    settings = MatchSettings(menu_mode, args.depth, time_limit, args.play_as - 1, args.ml_model, args.ml_sims,
                             bots=specs, move_delay=args.delay)
    if wall_filter is not None:
        settings.wall_filter = wall_filter
    run_gui(args.p1, args.p2, settings=settings, show_menu=mode is None)


if __name__ == "__main__":
    main()
