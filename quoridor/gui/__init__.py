"""Pygame GUI for Quoridor."""

from __future__ import annotations

import pygame

from ..game import Game
from .app import QuoridorGUI, RestartGame
from .layout import Layout
from .player import GuiHumanPlayer

__all__ = ["GuiHumanPlayer", "Layout", "QuoridorGUI", "RestartGame", "run_gui"]


def run_gui(name1: str = "Player 1", name2: str = "Player 2") -> None:
    """Open the window and play Human vs. Human matches until it is closed."""
    gui = QuoridorGUI()
    try:
        while True:
            gui.restart_requested = False
            game = Game(GuiHumanPlayer(name1, gui), GuiHumanPlayer(name2, gui), observers=[gui])
            game.run()
            if gui.quit_requested or not gui.restart_requested:
                break
    finally:
        pygame.quit()
