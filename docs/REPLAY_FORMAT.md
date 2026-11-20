# Replay file format and website handoff

This is the brief for building the replay website (PLAN.md steps 9, 9a, 9b and Vercel hosting).
The simulator and training code are done; the website only needs to read replay files.

## Where replays are

- Training writes `match_000001.json.gz`, `match_000100.json.gz`, `match_000200.json.gz`, ...
  (the very first match plus every 100th) to `/mnt/project-files/laser-tag/replays/`.
- For the site, copy them into `web/replays/` and keep a `web/replays/index.json` list
  (`[{"match": 1, "file": "match_000001.json.gz", "green": 5, "pink": -3, "winner": "green"}, ...]`).
  A small script that regenerates the index from the folder is welcome.
- Files are gzipped JSON, about 3-4 MB each. Decompress in the browser with
  `DecompressionStream("gzip")` (or pako). Serve them as plain binary so the browser doesn't
  try to decompress them itself (on Vercel, set `Content-Type: application/octet-stream` for `/replays/*.gz`).
- A sample file: `docs/sample_replay.json.gz` (match 1, untrained agents).

## Top level

| Key | Meaning |
|-----|---------|
| `version` | Format version (5: adds `duel_hints`; 4 added `aim_rewards`; 3 added `tags`; 2 added 60 FPS `motion`). |
| `match` | Match number. |
| `aim_rewards` | `true` for matches trained with Ken's aim rewards (match 701 onward): +1 when an opponent comes into an agent's sight, -1 when it comes into an opponent's sight. In the match picker, draw a divider labelled **"Aim rewards added"** between the last match without them and the first with them. Older files have no key (treat as `false`). |
| `duel_hints` | `true` for matches trained with the duel hints (match 2309 onward): extra training reward for the shooter on a tag, +7 if the target could see the shooter and hadn't fired first, +10 instead if the target's crosshair was also on the shooter. In the match picker, draw a divider labelled **"Hints added"** before the first match with them. Older files have no key (treat as `false`). |
| `trained_matches` | Matches the networks had trained on before this one. |
| `dt` | Seconds per step (0.1). |
| `steps` | Number of steps, `S` (1800 = 3 minutes). |
| `arena` | `width` 30, `depth` 20, `wall_height` 3, `boxes`: `[x0, y0, z0, x1, y1, z1, kind]` with kind in `wall`, `cover`, `platform`, `stair`. Units are metres; z is up. |
| `agents` | 8 entries: `{id, team: "green" or "pink", shape: "sphere" or "cube"}`. Ids 0-3 green, 4-7 pink. |
| `final_scores` | `{green, pink, per_agent: [8], winner: "green" / "pink" / "draw", aim_points_per_agent: [8], hint_points_per_agent: [8]}`. Scores are tag points only; aim points are training rewards shown separately. |
| `frames` | Per-step state at 10 Hz (the simulation rate), flat arrays, index `[step * 8 + agent]` (see below). |
| `motion` | Player movement at **60 FPS** for smooth playback (see below). |
| `shots` | List of `{step, shooter, target, tag, end}`. `target` is an agent id or -1 (hit a wall or nothing); `tag` is true when it scored; `end` is the beam end point `[x, y, z]`. The beam starts at the shooter's eye. |
| `tags` | Timeline markers, one per tag: `{step, frame, time_s, shooter, target, team, color}`. `team` is the team that **fired**, and `color` is that team's colour (green `#34c759`, pink `#ff5fa2`). `frame` is the 60 FPS frame. Ken wants every tag marked on the timeline scrubber in the shooting team's colour. |
| `network` | Network activations for the inspector panel (see below). |

## `frames`

All arrays are flat, row-major. `N = 8` agents.

| Key | Shape | Meaning |
|-----|-------|---------|
| `pos` | S x N x 3 | x, y, feet height (m). |
| `yaw` | S x N | Facing angle in radians, 0 = +x, counter-clockwise. |
| `pitch` | S x N | Gun/eye pitch in radians, + = up. |
| `crouch` | S x N | 1 if crouching. |
| `down` | S x N | 1 while tagged and waiting to respawn (hide or fade the player). |
| `gun_ready` | S x N | 1 if the gun is off cooldown (tip glows). |
| `radio` | S x N | Four-digit team radio code as a string, e.g. `"0427"`. |
| `team_scores` | S x 2 | Running green, pink score. |

Body geometry for drawing: radius 0.3 m, standing height 1.75 m (crouched 1.15 m),
eye height 1.6 m (crouched 1.0 m). Draw green as spheres and pink as cubes, roughly
body-sized, with a short gun barrel from chest height pointing along (yaw, pitch).

## `motion` (60 FPS playback)

Ken wants playback at 60 FPS. The simulation and the networks run at 10 steps per second,
so the recorder also saves player movement at 60 frames per second (6 frames per step):

| Key | Shape | Meaning |
|-----|-------|---------|
| `fps` | | 60. |
| `frames` | | Number of 60 FPS frames, `F = (steps - 1) * 6 + 1`. |
| `pos` | F x N x 3 | x, y, feet height. |
| `yaw` | F x N | Facing angle (already interpolated the short way round). |
| `pitch` | F x N | Gun pitch. |

Frame `f` lines up with simulation step `f / 6`. Drive the 3D scene from `motion` at 60 FPS
(times the playback speed), and look up everything else (`frames`, `shots`, `network`) at
step `floor(f / 6)`. Respawns are not interpolated, so players jump instead of gliding.

## `network` (inspector panel)

Each agent has its own network: vision (768 inputs) -> 64, other senses (153) -> 64,
both -> GRU memory (128) -> 58 action logits + 1 value.

Base64 arrays decode to uint8, shape `S x N x size`, dequantize with
`value = range[0] + byte / 255 * (range[1] - range[0])`.

| Key | Size | Meaning |
|-----|------|---------|
| `vision` | 8 rows x 24 cols per agent | What the agent sees. Each byte: `(byte >> 2) / 63 * 40` = distance in m, `byte & 3` = what it hit (0 nothing, 1 wall/floor, 2 teammate, 3 opponent). Row 0 is the top of the view, column 0 the left edge (90 x 60 degree view). |
| `enc_vision` | 64 | Vision encoder activations (range 0..4). |
| `enc_other` | 64 | Encoder for body, hearing, hit events and teammates' radio (range 0..4). |
| `memory` | 128 | GRU memory cells (range -1..1). Draw as a 16 x 8 grid. |
| `action_probs` | 58 | Probabilities per action head, in `heads` order: forward (back, none, fwd), strafe (left, none, right), turn (5 bins, hard left .. hard right), pitch (down, level, up), crouch (no, yes), fire (no, yes), then radio0..radio3 (10 each = probability of each digit). |
| `heads` | list | `{name, size}` for slicing `action_probs`. |
| `value` | S x N floats | The agent's own estimate of upcoming reward ("how well this is going"). |
| `policy_weights` | 8 x 128 x 58 | Memory -> action weights, int8 stored as uint8 + 128, multiply by `scale / 127`. Use `weight x memory activation` for the brightness of connection lines (show only the strongest few). |

The values at step `s` are the ones the network had when it chose the actions taken at step `s`.

## UI asks from Ken

- 3D view (Three.js): free orbit, top-down, and follow-agent first-person camera.
- Laser beams as short flashes in the team colour; tagged players blink/fade; scoreboard; timeline
  scrubber; speeds 0.5x to 8x; match picker; per-agent radio codes and a chart of the most-used codes.
- Network inspector panel (slide-out): pick one of 8 agents and watch its network update in sync with playback.
- Arena geometry all gray with simple shading so objects are easy to tell apart.
- Polished, following Apple's Human Interface Guidelines: system font, translucent blurred sidebars,
  8 pt grid, rounded corners, gentle spring animations, hover/press states, keyboard shortcuts
  (space, arrows, 1-8 for agents), light/dark mode, reduced-motion support.
- Static site in `web/`, deployed to Vercel from the GitHub repo.
