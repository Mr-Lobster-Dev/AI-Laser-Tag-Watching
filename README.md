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
pip install -r requirements.txt
python -m laser_tag.demo --map   # print the arena and play one match with random agents
python -m pytest tests           # rule tests
```

## Long training runs (GitHub Actions)

`.github/workflows/train.yml` trains in 5.5-hour runs that chain into each other (with a 6-hourly
schedule as a backup). Each run resumes from the `training-state` release (latest checkpoint and logs)
and uploads new replays and milestone checkpoints to the `replays` release. Start a run by hand from
the Actions tab with **Run workflow**. Repository variables: `TRAIN_UNTIL` (end time, default end of
2026-12-02 Bangkok) and `HINTS_UNTIL` (first match without the aim and duel hints).
