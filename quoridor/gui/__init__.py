"""Pygame GUI for Quoridor."""

from __future__ import annotations

from typing import Optional

import pygame

from ..bots import AlphaBetaBot, make_player
from ..game import Game
from .app import QuoridorGUI, RestartGame
from .layout import Layout
from .menu import MatchSettings, Menu
from .player import GuiBotPlayer, GuiHumanPlayer

__all__ = ["GuiBotPlayer", "GuiHumanPlayer", "Layout", "MatchSettings", "QuoridorGUI", "RestartGame", "run_gui"]


def build_players(settings: MatchSettings, gui: QuoridorGUI, names: tuple[str, str]):
    gui.bots_only = settings.mode == "bvb"
    gui.move_delay = settings.move_delay if gui.bots_only else 0.0
    gui.paused = False
    if settings.mode == "hvh":
        return GuiHumanPlayer(names[0], gui), GuiHumanPlayer(names[1], gui)
    if settings.mode == "bvb":
        return tuple(GuiBotPlayer(make_player(spec, f"Bot {i + 1}: {spec.label()}"), gui)
                     for i, spec in enumerate(settings.bots))
    if settings.ml_model:
        from ..ml.agent import MLPlayer

        inner = MLPlayer(settings.ml_model, "ML agent", settings.ml_sims)
    else:
        inner = AlphaBetaBot(f"Bot (depth {settings.depth})", settings.depth, settings.time_limit)
    bot = GuiBotPlayer(inner, gui)
    human = GuiHumanPlayer(names[settings.human_side], gui)
    return (human, bot) if settings.human_side == 0 else (bot, human)


def run_gui(name1: str = "Player 1", name2: str = "Player 2",
            settings: Optional[MatchSettings] = None, show_menu: bool = True) -> None:
    """Open the window; show the mode menu, then play until the window is closed.

    "New game" (R) returns to the menu with the previous choices preselected.
    """
    gui = QuoridorGUI()
    settings = settings or MatchSettings()
    try:
        while True:
            if show_menu:
                chosen = Menu(gui).run(settings)
                if chosen is None:
                    break
                settings = chosen
            show_menu = True
            gui.restart_requested = False
            p1, p2 = build_players(settings, gui, (name1, name2))
            game = Game(p1, p2, observers=[gui])
            game.run()
            if gui.quit_requested or not gui.restart_requested:
                break
    finally:
        pygame.quit()
