"""Computer players."""

from .alphabeta import MAX_DEPTH, AlphaBetaBot, search
from .factory import BotSpec, make_player, make_policy, parse_bot_spec

__all__ = ["MAX_DEPTH", "AlphaBetaBot", "BotSpec", "make_player", "make_policy", "parse_bot_spec", "search"]
