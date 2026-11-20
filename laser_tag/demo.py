"""Play one full match with random agents and print the result.

Usage: python -m laser_tag.demo [--seed N] [--map]
"""
import argparse
import time

import numpy as np

from .game import LaserTagGame, Action, MATCH_STEPS
from .arena import WIDTH, DEPTH, WALL, COVER, PLATFORM, STAIR


def ascii_map(game, res=0.5):
    chars = {WALL: "#", COVER: "c", PLATFORM: "P", STAIR: "s"}
    rows = []
    for y in np.arange(DEPTH - res / 2, 0, -res):
        row = ""
        for x in np.arange(res / 2, WIDTH, res):
            ch = "."
            for (lo, hi, kind) in game.arena.boxes:
                if kind in chars and lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
                    ch = chars[kind] if chars.get(kind) != "s" or ch == "." else ch
                    if kind in (WALL, PLATFORM):
                        break
            for i in range(game.n_agents):
                if abs(game.pos[i, 0] - x) < res / 2 and abs(game.pos[i, 1] - y) < res / 2:
                    ch = "G" if game.team[i] == 0 else "K"
            row += ch
        rows.append(row)
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", action="store_true", help="print the arena layout")
    args = ap.parse_args()
    game = LaserTagGame(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    if args.map:
        print(ascii_map(game))
        print("# wall  c cover (1 m)  P platform (1.5 m)  s stairs  G green  K pink\n")
    t0 = time.time()
    shots = 0
    for _ in range(MATCH_STEPS):
        acts = [Action(forward=int(rng.integers(-1, 2)), strafe=int(rng.integers(-1, 2)),
                       turn=float(rng.uniform(-1, 1)), pitch=float(rng.uniform(-0.2, 0.2)),
                       crouch=bool(rng.random() < 0.1), fire=bool(rng.random() < 0.2))
                for _ in range(game.n_agents)]
        _, _, done, info = game.step(acts)
        shots += len(info["shots"])
        if done:
            break
    g, p = game.team_scores()
    print(f"Match over in {time.time() - t0:.1f}s wall time. Shots fired: {shots}")
    print(f"Green {g:+.0f}  Pink {p:+.0f}  -> winner: {game.winner()}")


if __name__ == "__main__":
    main()
