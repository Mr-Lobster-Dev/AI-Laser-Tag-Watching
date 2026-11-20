# AI Laser Tag

Two teams of four AI agents (green and pink) learn laser tag with reinforcement learning
in a simple 3D arena. See [PLAN.md](PLAN.md) for the full plan.

## Status
The arena and match rules are built. The neural networks, training and replay website come next.

## Arena
- 30 m x 20 m, 3 m walls, built from boxes: full-height walls, waist-high (1 m) cover,
  two 1.5 m platforms with stairs, and a 1.5 m central tower with stairs on both sides.
- Point-symmetric, so neither team has a layout advantage.
- No bases: each team respawns at a random spot on its own end, so agents can pick any favourite position.

## Rules
3-minute matches at 10 steps per second. Lasers have unlimited range and stop at walls (no bounces).
5 s cooldown. +5 for tagging an opponent, -3 for being tagged, friendly fire does nothing.
A tagged agent is out for 3 s, then respawns. Higher team score wins.

Agents perceive only a first-person 90° x 60° view (24 x 8 rays that walls block) plus their own state.

## Try it
```
pip install numpy pytest
python -m laser_tag.demo --map   # print the arena and play one match with random agents
python -m pytest tests           # rule tests
```
