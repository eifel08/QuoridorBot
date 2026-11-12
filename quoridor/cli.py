"""Terminal front-end: text view + launcher (Human vs Human or Human vs Bot)."""

from __future__ import annotations

from typing import Callable, Optional

from .game import Game, GameObserver
from .moves import Move
from .player import HELP_TEXT, HumanPlayer
from .renderer import PAWN_SYMBOLS, render_board, render_status
from .state import BoardState


class TextView(GameObserver):
    """Prints the board and game events to the terminal."""

    def __init__(self, output_func: Callable[[str], None] = print) -> None:
        self._out = output_func

    def _show(self, game: Game, state: BoardState) -> None:
        self._out("")
        self._out(render_board(state))
        self._out(render_status(state, game.names))

    def on_turn_start(self, game: Game, state: BoardState) -> None:
        self._show(game, state)
        player = game.players[state.current_player]
        hint = " (type 'help' for commands)" if isinstance(player, HumanPlayer) else ""
        self._out(f"\n{player.name}'s turn [{PAWN_SYMBOLS[state.current_player]}]{hint}")

    def on_move(self, game: Game, player: int, move: Move, before: BoardState, after: BoardState) -> None:
        self._out(f"{game.players[player].name} played {move}.")

    def on_game_over(self, game: Game, state: BoardState) -> None:
        self._show(game, state)
        self._out(f"\n*** {game.players[state.winner].name} wins! ***")

    def on_abort(self, game: Game) -> None:
        self._out("\nGame aborted.")


def _ask(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def _ask_depth(prompt: str, default: str) -> int:
    while True:
        try:
            depth = int(_ask(prompt, default))
            if 1 <= depth <= 30:
                return depth
        except ValueError:
            pass
        print("  ! Please enter a whole number from 1 to 30.")


class DelayView(GameObserver):
    """Pauses after each move so a Bot vs Bot game can be followed."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def on_move(self, game: Game, player: int, move: Move, before: BoardState, after: BoardState) -> None:
        if self.seconds > 0:
            import time

            time.sleep(self.seconds)


def _ask_walls(prompt: str) -> int:
    """Which walls the bot's search considers; returns the engine's wall_filter value."""
    from .bots.cpp_engine import WALL_MODES

    menu = ", ".join(f"{i} = {label}" for i, (_, _, label) in enumerate(WALL_MODES, 1))
    while True:
        answer = _ask(f"{prompt}: {menu}", "2")
        if answer in ("1", "2", "3"):
            return WALL_MODES[int(answer) - 1][1]
        print("  ! Please enter 1, 2 or 3.")


def ask_match_settings() -> dict:
    """Interactive mode/depth questions. Returns keyword args for run_cli."""
    mode = _ask("Game mode: 1 = Human vs Human, 2 = Human vs Bot, 3 = Bot vs Bot", "1")
    if mode == "3":
        from .bots import BotSpec

        d1 = _ask_depth("Bot 1 (Player 1, moves first) search depth (1-30)", "2")
        w1 = _ask_walls("Bot 1 walls")
        d2 = _ask_depth("Bot 2 (Player 2) search depth (1-30)", "4")
        w2 = _ask_walls("Bot 2 walls")
        try:
            delay = float(_ask("Pause between moves in seconds", "0.5"))
        except ValueError:
            delay = 0.5
        specs = (BotSpec(depth=d1).with_wall_filter(w1), BotSpec(depth=d2).with_wall_filter(w2))
        return {"mode": "bvb", "bot_specs": specs, "move_delay": delay}
    if mode != "2":
        return {"mode": "hvh"}
    depth = _ask_depth("Bot search depth (1-30)", "4")
    t = _ask("Think time limit per bot move in seconds (0 = no limit)", "5")
    try:
        time_limit = float(t) or None
    except ValueError:
        time_limit = 5.0
    walls = _ask_walls("Walls the bot considers")
    side = _ask("Play as 1 = Player 1 (moves first) or 2 = Player 2", "1")
    return {"mode": "hvb", "depth": depth, "time_limit": time_limit, "human_side": 1 if side == "2" else 0,
            "wall_filter": walls}


class BotInfoView(GameObserver):
    """Prints a one-line summary after each bot move."""

    def on_move(self, game: Game, player: int, move: Move, before: BoardState, after: BoardState) -> None:
        info = getattr(game.players[player], "last_info", None)
        if info and "nodes" in info:
            from .evaluation import Eval

            ev = Eval.from_info(info, player)
            outcome = f"eval {ev.text()}" if ev is not None else f"forced {info.get('forced')}"
            print(f"  (searched depth {info['depth_reached']}/{info['depth_requested']}, "
                  f"{info['nodes']:,} positions in {info['seconds']:.2f}s, {outcome})")


def run_cli(mode: Optional[str] = None, depth: int = 4, time_limit: Optional[float] = 5.0,
            human_side: int = 0, ml_model: Optional[str] = None, ml_sims: int = 64,
            bot_specs=None, move_delay: float = 0.5, wall_filter: Optional[int] = None) -> None:
    from .bots import BotSpec, make_player, make_search_bot

    print("=== Quoridor ===")
    print("Player 1 [1] starts at the bottom (row 8) and must reach row 0.")
    print("Player 2 [2] starts at the top (row 0) and must reach row 8.\n")
    try:
        if mode is None:
            opts = ask_match_settings()
            mode = opts["mode"]
            depth = opts.get("depth", depth)
            time_limit = opts.get("time_limit", time_limit)
            human_side = opts.get("human_side", human_side)
            bot_specs = opts.get("bot_specs", bot_specs)
            move_delay = opts.get("move_delay", move_delay)
            wall_filter = opts.get("wall_filter", wall_filter)
        print()
        if mode == "bvb":
            specs = bot_specs or (BotSpec(depth=2), BotSpec(depth=4))
            players = tuple(make_player(sp, f"Bot {i + 1} ({sp.label()})") for i, sp in enumerate(specs))
            print(f"Bot vs Bot: {players[0].name} vs {players[1].name}. Ctrl+C stops.")
        elif mode == "hvb":
            print(HELP_TEXT)
            name = input("\nYour name (blank = 'You'): ").strip() or "You"
            human = HumanPlayer(name)
            if ml_model:
                from .ml.agent import MLPlayer

                bot = MLPlayer(ml_model, "ML agent", ml_sims)
            else:
                options = {} if wall_filter is None else {"wall_filter": wall_filter}
                bot = make_search_bot(f"Bot (depth {depth})", depth, time_limit, options)
            players = (human, bot) if human_side == 0 else (bot, human)
        else:
            print(HELP_TEXT)
            name1 = input("\nName for Player 1 (blank = 'Player 1'): ").strip() or "Player 1"
            name2 = input("Name for Player 2 (blank = 'Player 2'): ").strip() or "Player 2"
            players = (HumanPlayer(name1), HumanPlayer(name2))
    except (EOFError, KeyboardInterrupt):
        print()
        return
    observers = [TextView(), BotInfoView()]
    if mode == "bvb":
        observers.append(DelayView(move_delay))
    game = Game(*players, observers=observers)
    try:
        game.run()
    except KeyboardInterrupt:
        print("\nGame aborted.")
