"""Terminal front-end: text view + Human vs. Human launcher."""

from __future__ import annotations

from typing import Callable

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


def run_cli() -> None:
    print("=== Quoridor ===")
    print("Player 1 [1] starts at the bottom (row 8) and must reach row 0.")
    print("Player 2 [2] starts at the top (row 0) and must reach row 8.\n")
    print(HELP_TEXT)
    try:
        name1 = input("\nName for Player 1 (blank = 'Player 1'): ").strip() or "Player 1"
        name2 = input("Name for Player 2 (blank = 'Player 2'): ").strip() or "Player 2"
    except (EOFError, KeyboardInterrupt):
        print()
        return
    game = Game(HumanPlayer(name1), HumanPlayer(name2), observers=[TextView()])
    try:
        game.run()
    except KeyboardInterrupt:
        print("\nGame aborted.")
