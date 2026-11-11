"""Entry point: Human vs. Human Quoridor in the terminal."""

from quoridor import Game, HumanPlayer
from quoridor.player import HELP_TEXT


def main() -> None:
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
    game = Game(HumanPlayer(name1), HumanPlayer(name2))
    try:
        game.run()
    except KeyboardInterrupt:
        print("\nGame aborted.")


if __name__ == "__main__":
    main()
