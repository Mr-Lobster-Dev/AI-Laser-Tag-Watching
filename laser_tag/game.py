"""Laser tag match rules: 2 teams x 4 agents, 3-minute matches at 10 steps/s.

Rules (from the project plan):
- Lasers are instant rays with unlimited range, stopped by geometry, no bounces.
- 5 s cooldown between shots. Hitting an opponent: +5. Being hit: -3.
  Hitting a teammate does nothing.
- A tagged agent is down for 3 s, then respawns at a random spot on its team's end of the arena.
  There are no bases or objectives: agents are free to hold any spot they like.
- Team score = sum of its members' rewards; the higher team score wins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arena import Arena, ray_boxes

DT = 0.1
MATCH_STEPS = 1800            # 3 minutes
COOLDOWN_STEPS = 50           # 5 seconds
RESPAWN_STEPS = 30            # 3 seconds
HIT_REWARD, TAGGED_REWARD = 5.0, -3.0
WALK_SPEED, TURN_SPEED, PITCH_SPEED = 3.0, np.radians(90), np.radians(60)
RADIUS = 0.3                  # same body size for both teams
STAND_H, CROUCH_H = 1.75, 1.15
EYE_STAND, EYE_CROUCH = 1.6, 1.0
FOV_H, FOV_V = np.radians(90), np.radians(60)
VISION_W, VISION_H = 24, 8
MAX_VIEW = 40.0

GREEN, PINK = 0, 1
TEAM_NAMES = ("green", "pink")
# Ray-hit codes in vision observations.
SEE_NOTHING, SEE_WORLD, SEE_TEAMMATE, SEE_OPPONENT = 0, 1, 2, 3


@dataclass
class Action:
    forward: int = 0      # -1 back, 0, +1 forward
    strafe: int = 0       # -1 left, 0, +1 right
    turn: float = 0.0     # -1..1 (yaw rate fraction, + = left/counter-clockwise)
    pitch: float = 0.0    # -1..1
    crouch: bool = False
    fire: bool = False


class LaserTagGame:
    n_agents = 8

    def __init__(self, seed: int | None = None):
        self.arena = Arena()
        self.rng = np.random.default_rng(seed)
        self.team = np.array([GREEN] * 4 + [PINK] * 4)
        self.reset()

    # ---------- state ----------
    def reset(self):
        n = self.n_agents
        self.pos = np.zeros((n, 3))           # x, y, feet height
        self.yaw = np.zeros(n)
        self.pitch = np.zeros(n)
        self.crouch = np.zeros(n, dtype=bool)
        self.cooldown = np.zeros(n, dtype=int)
        self.down = np.zeros(n, dtype=int)    # steps until respawn; 0 = active
        self.score = np.zeros(n)
        self.step_count = 0
        for i in range(n):
            self._spawn(i)
        return self.observe_all()

    def _spawn(self, i):
        x, y, yaw = self.arena.random_spawn(self.team[i], self.rng)
        self.pos[i] = (x, y, 0.0)
        self.yaw[i] = yaw
        self.pitch[i] = 0.0
        self.crouch[i] = False
        self.cooldown[i] = 0

    def eye(self, i):
        x, y, feet = self.pos[i]
        return np.array([x, y, feet + (EYE_CROUCH if self.crouch[i] else EYE_STAND)])

    def aim(self, i):
        cp = np.cos(self.pitch[i])
        return np.array([cp * np.cos(self.yaw[i]), cp * np.sin(self.yaw[i]), np.sin(self.pitch[i])])

    def hitboxes(self):
        """Axis-aligned hitboxes of active agents: (lo, hi, agent ids)."""
        ids = np.flatnonzero(self.down == 0)
        p = self.pos[ids]
        h = np.where(self.crouch[ids], CROUCH_H, STAND_H)
        lo = np.stack([p[:, 0] - RADIUS, p[:, 1] - RADIUS, p[:, 2]], 1)
        hi = np.stack([p[:, 0] + RADIUS, p[:, 1] + RADIUS, p[:, 2] + h], 1)
        return lo, hi, ids

    def trace(self, origins, dirs, ignore):
        """First thing each ray hits. Returns (distance, agent id or -1 for world)."""
        t_world, _ = self.arena.raycast(origins, dirs)
        lo, hi, ids = self.hitboxes()
        keep = ids != ignore
        lo, hi, ids = lo[keep], hi[keep], ids[keep]
        if len(ids):
            t_agent, k = ray_boxes(origins, dirs, lo, hi)
            agent_first = t_agent < t_world
            who = np.where(agent_first, ids[np.maximum(k, 0)], -1)
            return np.where(agent_first, t_agent, t_world), who
        return t_world, np.full(len(t_world), -1)

    # ---------- rules ----------
    def step(self, actions: list[Action]):
        """Advance one tick (0.1 s). Returns (observations, rewards, done, info)."""
        n = self.n_agents
        rewards = np.zeros(n)
        hits = []
        active = self.down == 0

        # Movement and aiming.
        for i in range(n):
            if not active[i]:
                continue
            a = actions[i]
            self.crouch[i] = bool(a.crouch)
            self.yaw[i] = (self.yaw[i] + np.clip(a.turn, -1, 1) * TURN_SPEED * DT) % (2 * np.pi)
            self.pitch[i] = np.clip(self.pitch[i] + np.clip(a.pitch, -1, 1) * PITCH_SPEED * DT,
                                    -np.radians(60), np.radians(60))
            f, s = np.clip(a.forward, -1, 1), np.clip(a.strafe, -1, 1)
            if f or s:
                c, sn = np.cos(self.yaw[i]), np.sin(self.yaw[i])
                d = np.array([f * c + s * sn, f * sn - s * c])
                d = d / np.linalg.norm(d) * WALK_SPEED * DT * (0.5 if self.crouch[i] else 1.0)
                x, y, feet = self.pos[i]
                h = CROUCH_H if self.crouch[i] else STAND_H
                # Try full move, then slide along each axis.
                for nx, ny in ((x + d[0], y + d[1]), (x + d[0], y), (x, y + d[1])):
                    r = self.arena.try_move(x, y, feet, nx, ny, RADIUS, h)
                    if (r[0], r[1]) != (x, y):
                        break
                self.pos[i] = r

        # Firing: all shots this tick resolve against positions after movement.
        shooters = [i for i in range(n) if active[i] and actions[i].fire and self.cooldown[i] == 0]
        tagged = set()
        for i in shooters:
            self.cooldown[i] = COOLDOWN_STEPS
            dist, who = self.trace(self.eye(i)[None], self.aim(i)[None], ignore=i)
            j = int(who[0])
            hits.append({"shooter": i, "target": j, "distance": float(dist[0])})
            if j >= 0 and self.team[j] != self.team[i] and j not in tagged:
                rewards[i] += HIT_REWARD
                rewards[j] += TAGGED_REWARD
                tagged.add(j)
        for j in tagged:
            self.down[j] = RESPAWN_STEPS

        # Timers and respawns.
        self.cooldown = np.maximum(self.cooldown - 1, 0)
        for j in range(n):
            if self.down[j] > 0 and j not in tagged:
                self.down[j] -= 1
                if self.down[j] == 0:
                    self._spawn(j)

        self.score += rewards
        self.step_count += 1
        done = self.step_count >= MATCH_STEPS
        return self.observe_all(), rewards, done, {"shots": hits, "team_scores": self.team_scores()}

    def team_scores(self):
        return [float(self.score[self.team == t].sum()) for t in (GREEN, PINK)]

    def winner(self):
        g, p = self.team_scores()
        return "draw" if g == p else TEAM_NAMES[int(p > g)]

    # ---------- perception ----------
    def observe(self, i):
        """What agent i can perceive: a first-person ray grid plus its own body state.

        vision: (VISION_H, VISION_W, 2) = [distance / MAX_VIEW, hit code]. Walls block
        sight; nothing outside the 90 x 60 degree view cone is visible.
        """
        yaws = self.yaw[i] + np.linspace(FOV_H / 2, -FOV_H / 2, VISION_W)
        pitches = self.pitch[i] + np.linspace(FOV_V / 2, -FOV_V / 2, VISION_H)
        P, Y = np.meshgrid(pitches, yaws, indexing="ij")
        dirs = np.stack([np.cos(P) * np.cos(Y), np.cos(P) * np.sin(Y), np.sin(P)], -1).reshape(-1, 3)
        origins = np.repeat(self.eye(i)[None], len(dirs), 0)
        dist, who = self.trace(origins, dirs, ignore=i)
        code = np.where(who < 0, np.where(np.isfinite(dist) & (dist <= MAX_VIEW), SEE_WORLD, SEE_NOTHING),
                        np.where(self.team[np.maximum(who, 0)] == self.team[i], SEE_TEAMMATE, SEE_OPPONENT))
        d = np.minimum(np.nan_to_num(dist, posinf=MAX_VIEW), MAX_VIEW) / MAX_VIEW
        vision = np.stack([d, code], -1).reshape(VISION_H, VISION_W, 2)
        body = np.array([
            self.cooldown[i] / COOLDOWN_STEPS,
            self.down[i] / RESPAWN_STEPS,
            float(self.crouch[i]),
            self.pos[i, 2] / 1.5,
            self.pitch[i] / np.radians(60),
            1.0 - self.step_count / MATCH_STEPS,
        ])
        return {"vision": vision, "body": body}

    def observe_all(self):
        return [self.observe(i) for i in range(self.n_agents)]
