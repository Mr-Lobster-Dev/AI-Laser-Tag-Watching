# AI Laser Tag: Execution Plan

Status: plan only. Execution starts 2026-11-20 11:00 Asia/Bangkok.

## Constraints found
- Workspace: 4 CPU cores, 15 GB RAM, no GPU, Python 3.13. Everything must be CPU-cheap.
- The cloud container is ephemeral. Code, checkpoints and replays live in `/mnt/project-files/laser-tag/` (and a GitHub repo if Ken attaches one), never only in the container.
- Sessions can be reclaimed when idle, so training is resumable from the latest checkpoint.

## 1. Tech stack
- Python + NumPy (+ Numba for raycasting) for the simulator. No game engine: Unity/Godot are too heavy for 4 CPUs.
- PyTorch (CPU) for the networks and training.
- Replay site: static HTML + Three.js in the repo's `web/` folder, deployed to Vercel from the GitHub repo once it's built.

## 2. Arena
- Realistic size: about 30 m x 20 m, 3 m ceiling, built from axis-aligned boxes (walls, pillars, crates).
- Vantage points: two raised platforms (1.5 m) reachable by ramps, a central tower, low cover (1 m) you can shoot over while crouched or standing, and a few long sightlines.
- Point-symmetric so neither team has a layout advantage. No bases or objectives: each team respawns at a random spot along its own end of the arena, so agents are free to find and hold a favourite spot.
- Physics: 10 decisions per second, walk speed 3 m/s, crouch, no jumping in v1. 3 minutes = 1,800 steps.
- Lasers: instant ray, unlimited range, stopped by geometry, no bounce. 5 s cooldown (50 steps). A hit on a teammate does nothing; hitting yourself is impossible.

## 2a. Teams and players
- **Green team:** spheres. **Pink team:** cubes.
- Shapes are visual only. Both use the same hitbox (same width and height) so a cube's corners don't make pink easier to hit.
- Each player carries a simple laser gun: a short barrel that points exactly where the player aims (yaw and pitch), with a small glow on the tip that dims during the 5 s cooldown and brightens when ready. Shots show as a beam flash in the team colour.
- The gun also matters for learning: a person can see where an opponent is pointing, so an agent that sees another player also gets that player's rough facing direction and whether its gun is glowing (ready). This lets them learn to dodge an aimed gun or rush one that's cooling down.

## 3. Perception (human-equivalent)
Each agent sees only what a person in its position could:
- **Vision:** a 90° x 60° field of view sampled as a low-res ray grid (24 x 8 rays). Each ray returns distance plus what it hit (wall/floor, teammate, opponent). For a visible player it also returns their facing direction relative to the viewer and whether their gun is ready. Nothing behind walls, nothing outside the view cone.
- **Hearing:** footsteps and laser shots within ~15 m, as a coarse direction (8 sectors) and loudness. No exact positions.
- **Body sense:** own cooldown remaining, own velocity, crouch state, height, time left in match.
- **Events:** "I was hit" (with rough direction, like a vest buzzing), "I scored a hit".
- **No** global map, enemy coordinates, or scoreboard beyond what a vest display would show (own team score).

## 4. Actions
Per step, multi-discrete: move forward/back/none, strafe left/right/none, turn yaw (5 bins), tilt pitch (3 bins), crouch toggle, fire (ignored while cooling down), plus a communication output (below).

## 5. Team communication
- Each agent emits a "radio" message every step: 4 discrete symbols from a 16-word vocabulary that the agents invent themselves.
- Teammates receive all three teammates' last messages as input. Opponents never hear them.
- Nothing about the meaning is hard-coded; the language emerges from training. Replays show each message so Ken can watch a vocabulary form.

## 6. Network and learning
- **Network per agent (~120k params):** ray grid -> small conv/MLP encoder; hearing + body + radio -> MLP; concatenate -> GRU (128) for memory -> policy heads + value head.
- **Eight separate networks**, one per agent, same architecture, own weights, so each can develop its own style.
- **Algorithm:** PPO with a recurrent policy (independent PPO per agent). Optional upgrade: centralized critic (MAPPO) that sees the full state during training only; agents still act from their own perception.
- **Rewards exactly as specified:** +5 for tagging an opponent, -3 for being tagged. Team score = sum of members. Winning team = higher score. Small optional shaping (e.g. tiny penalty for walking into walls) only if learning stalls, and it would be flagged.
- **Respawn rule (assumption, needs Ken's ok):** a tagged agent is "down" for 3 s then respawns at a random spot on its own end of the arena. Otherwise a match would end early.

## 7. Match runner
- Runs N matches in parallel worker processes (3 workers on 4 cores), collects trajectories, runs a PPO update every few matches.
- Throughput target: about 10 matches per minute on CPU, so every-100th-match replays appear roughly every 10 minutes.
- Checkpoint all 8 networks every 50 matches to `/mnt/project-files/laser-tag/checkpoints/`; resume automatically.
- Training stats (scores, hit rate, message entropy) logged to a CSV.

## 8. Recording every 100th match
- Matches 1, 100, 200, ... are saved as compact replays: per step, each agent's position, yaw, pitch, crouch, fire event, hit events, radio symbols, and both scores.
- About 1,800 steps x 8 agents, gzipped JSON, roughly 200-400 KB each, saved to `/mnt/project-files/laser-tag/replays/`.

## 9. Replay website
- Three.js 3D view of the arena: free-orbit camera, top-down view, and "follow agent" first-person view.
- Laser beams drawn as flashes, tagged agents blink, live scoreboard, timeline scrubber, 0.5x to 8x speed.
- Sidebar: match picker (every 100th match), per-agent radio messages, and a training progress chart.
- **Hosting: Vercel.** Once the site works, connect the GitHub repo to a Vercel project (static site, root `web/`). Each push to the main branch redeploys automatically. Nothing is deployed until the site is built and checked.
- New replays (3-5 MB each) are committed to `web/replays/` as training goes on, which triggers a redeploy. If the repo gets too big (over ~1 GB), they move to Vercel Blob storage and the site loads them from there.

### 9a. Neural network inspector panel
- A slide-out panel next to the 3D view. Pick any of the 8 agents and watch its network's values move in time with the replay. Scrubbing and speed changes stay in sync.
- It shows the whole flow, left to right, as a live diagram:
  - **Inputs:** the agent's 24 x 8 vision grid drawn as a small heatmap (what it "sees"), hearing sectors, cooldown, and incoming radio symbols.
  - **Encoder layer:** 64 activations as a glowing node strip.
  - **Memory (GRU, 128 units):** a 16 x 8 grid of cells whose colour shows each value, so you can watch memory flip when an enemy appears.
  - **Outputs:** probability bars for each action (move, turn, fire), the radio symbols it chose, and the value estimate ("how well I think this is going") as a small line chart.
  - Connection lines between layers, brightness scaled by weight x activation, limited to the strongest links so it stays readable.
- **Data:** the recorder also saves these activations for every 100th match, quantized to 8 bits. That makes each replay roughly 3-5 MB, which is fine.
- **"Real time" here means synced to replay playback.** A static site can't stream values from training live. A live training view would need a running server, which is possible later if Ken wants it.

### 9b. UI polish (Apple Human Interface Guidelines)
- Follow HIG principles: clarity, deference (the arena is the content, chrome stays quiet), and depth.
- Details: system font stack (SF Pro on Apple devices), translucent "material" sidebars with background blur, 8 pt spacing grid, continuous rounded corners, SF-style monochrome icons, a segmented control for camera modes, a native-feeling scrubber, gentle spring animations on panel open/close, hover and press states on every control, keyboard shortcuts (space, arrows, 1-8 for agents), automatic light/dark mode, reduced-motion support, and good contrast.
- No "Apple HIG" skill is installed in this workspace, so I'll apply the guidelines directly when building the UI. If Ken has the skill, enabling it before 11:00 lets me use it instead.

## 10. Testing
- Unit tests: raycast hits/occlusion, no wall bounce, cooldown timing, friendly fire no-op, scoring (+5/-3), 3-minute timer, perception never leaks hidden opponents.
- Smoke test: random agents play one full match, replay loads in the site.
- Learning check: after ~500 matches, hit rate must beat random agents; if not, debug before running long.

## 11. Order of work and time estimates
| # | Step | Estimate |
|---|------|----------|
| 1 | Simulator: arena, movement, lasers, scoring + unit tests | 1.5 h |
| 2 | Perception + comms + Gym-style multi-agent API | 1 h |
| 3 | Networks, PPO trainer, parallel match runner, checkpoints | 1.5 h |
| 4 | Replay recording (incl. activations) + replay website, verified with random agents | 2 h |
| 4b | Network inspector panel + HIG polish pass | 1.5 h |
| 5 | Short training run, check learning, tune | 1 h |
| 6 | Long training run (background, resumable) | hours to days |

Training (step 6) can start right after step 3 and run in the background while the website is built.

First watchable replays: about 6-7 hours after start. Visible tactics (taking cover, using high ground, coordinated pushes) may take a day or more of CPU training, and are not guaranteed.

## Open questions for Ken
1. Respawn after 3 s when tagged (recommended), or eliminated for the rest of the match?
2. Should code go in a GitHub repo? If so, create one and attach it; otherwise it lives in project files.
