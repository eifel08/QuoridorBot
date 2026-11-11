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
        name = game.players[state.current_player].name
        self._out(f"\n{name}'s turn [{PAWN_SYMBOLS[state.current_player]}] (type 'help' for commands)")

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


def ask_match_settings() -> dict:
    """Interactive mode/depth questions. Returns keyword args for run_cli."""
    mode = _ask("Game mode: 1 = Human vs Human, 2 = Human vs Bot", "1")
    if mode != "2":
        return {"mode": "hvh"}
    while True:
        try:
            depth = int(_ask("Bot search depth (1-30)", "4"))
            if 1 <= depth <= 30:
                break
        except ValueError:
            pass
        print("  ! Please enter a whole number from 1 to 30.")
    t = _ask("Think time limit per bot move in seconds (0 = no limit)", "5")
    try:
        time_limit = float(t) or None
    except ValueError:
        time_limit = 5.0
    side = _ask("Play as 1 = Player 1 (moves first) or 2 = Player 2", "1")
    return {"mode": "hvb", "depth": depth, "time_limit": time_limit, "human_side": 1 if side == "2" else 0}


class BotInfoView(GameObserver):
    """Prints a one-line summary after each bot move."""

    def on_move(self, game: Game, player: int, move: Move, before: BoardState, after: BoardState) -> None:
        info = getattr(game.players[player], "last_info", None)
        if info and "nodes" in info:
            outcome = f"forced {info['forced']}" if info.get("forced") else f"score {info['score']:+d}"
            print(f"  (searched depth {info['depth_reached']}/{info['depth_requested']}, "
                  f"{info['nodes']:,} positions in {info['seconds']:.2f}s, {outcome})")


def run_cli(mode: Optional[str] = None, depth: int = 4, time_limit: Optional[float] = 5.0,
            human_side: int = 0, ml_model: Optional[str] = None, ml_sims: int = 64) -> None:
    from .bots import AlphaBetaBot

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
        print()
        print(HELP_TEXT)
        if mode == "hvb":
            name = input("\nYour name (blank = 'You'): ").strip() or "You"
            human = HumanPlayer(name)
            if ml_model:
                from .ml.agent import MLPlayer

                bot = MLPlayer(ml_model, "ML agent", ml_sims)
            else:
                bot = AlphaBetaBot(f"Bot (depth {depth})", depth, time_limit)
            players = (human, bot) if human_side == 0 else (bot, human)
        else:
            name1 = input("\nName for Player 1 (blank = 'Player 1'): ").strip() or "Player 1"
            name2 = input("Name for Player 2 (blank = 'Player 2'): ").strip() or "Player 2"
            players = (HumanPlayer(name1), HumanPlayer(name2))
    except (EOFError, KeyboardInterrupt):
        print()
        return
    game = Game(*players, observers=[TextView(), BotInfoView()])
    try:
        game.run()
    except KeyboardInterrupt:
        print("\nGame aborted.")
