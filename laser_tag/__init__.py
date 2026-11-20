"""AI laser tag: a small 3D arena for reinforcement learning agents."""
from .arena import Arena
from .game import LaserTagGame, Action

__all__ = ["Arena", "LaserTagGame", "Action"]
