"""Laser tag match rules: 2 teams x 4 agents, 3-minute matches at 10 steps/s.

Rules (from the project plan):
- Green team players are spheres, pink team players are cubes. Shapes are visual
  only: both use the same hitbox.
- Every player carries a laser gun whose barrel points where it aims. The tip glows
  when the gun is ready. Other players can see that aim and glow.
- Lasers are instant rays with unlimited range, stopped by geometry, no bounces.
- 5 s cooldown between shots. Hitting an opponent: +5. Being hit: -3.
  Hitting a teammate does nothing.
- A tagged agent is down for 3 s, then respawns at a random spot on its team's end
  of the arena. There are no bases or objectives: agents can hold any spot they like.
- Team radio: every step each agent sends a four-digit code (each digit 0-9) that
  only its teammates receive, one step later.
- Team score = sum of its members' tag points; the higher team score wins.
- Optional aim rewards (training from match 701 on): +1 when an opponent comes into
  your sight, -1 when you come into an opponent's sight. "In sight" means inside the
  90 x 60 degree view with a clear line of sight. Each viewer/opponent pair can
  trigger again only after 3 s, so flicking your view back and forth doesn't farm
  points. Aim rewards are training rewards only; the scoreboard stays tag points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arena import Arena, trace_rays

DT = 0.1
MATCH_STEPS = 1800            # 3 minutes
COOLDOWN_STEPS = 50           # 5 seconds
RESPAWN_STEPS = 30            # 3 seconds
HIT_REWARD, TAGGED_REWARD = 5.0, -3.0
SIGHT_REWARD, SEEN_PENALTY = 1.0, -1.0
SIGHT_RETRIGGER_STEPS = 30    # 3 seconds per viewer/opponent pair
WALK_SPEED, TURN_SPEED, PITCH_SPEED = 3.0, np.radians(90), np.radians(60)
MAX_PITCH = np.radians(60)
RADIUS = 0.3                  # same body size for both teams
STAND_H, CROUCH_H = 1.75, 1.15
EYE_STAND, EYE_CROUCH = 1.6, 1.0
FOV_H, FOV_V = np.radians(90), np.radians(60)
VISION_W, VISION_H = 24, 8
MAX_VIEW = 40.0
HEARING_RANGE = 15.0
SECTORS = 8                   # hearing / hit-direction resolution (45 degrees each)
RADIO_DIGITS, RADIO_BASE = 4, 10

GREEN, PINK = 0, 1
TEAM_NAMES = ("green", "pink")
TEAM_SHAPES = ("sphere", "cube")
# Ray-hit codes in vision observations.
SEE_NOTHING, SEE_WORLD, SEE_TEAMMATE, SEE_OPPONENT = 0, 1, 2, 3

# Observation sizes (used by the networks).
VISION_CHANNELS = 4           # distance, hit code / 3, aimed-at-me, gun ready
BODY_SIZE = 7
HEARING_SIZE = 2 * SECTORS    # footsteps, gunshots
EVENT_SIZE = 2 + SECTORS      # was hit, scored a hit, hit direction
RADIO_IN_SIZE = 3 * RADIO_DIGITS * RADIO_BASE   # 3 teammates x one-hot digits


@dataclass
class Action:
    forward: int = 0      # -1 back, 0, +1 forward
    strafe: int = 0       # -1 left, 0, +1 right
    turn: float = 0.0     # -1..1 (yaw rate fraction, + = left/counter-clockwise)
    pitch: float = 0.0    # -1..1
    crouch: bool = False
    fire: bool = False
    radio: tuple = (0, 0, 0, 0)   # four digits, each 0-9


IDLE = Action()


def _sector(angle):
    """Map a relative angle (radians, 0 = straight ahead, + = left) to 0..SECTORS-1."""
    return (np.floor(((angle + np.pi / SECTORS) % (2 * np.pi)) / (2 * np.pi / SECTORS))).astype(int) % SECTORS


class LaserTagGame:
    n_agents = 8

    def __init__(self, seed: int | None = None, aim_rewards: bool = False):
        self.aim_rewards = aim_rewards
        self.arena = Arena()
        self.rng = np.random.default_rng(seed)
        self.team = np.array([GREEN] * 4 + [PINK] * 4)
        self.teammates = np.array([[j for j in range(8) if self.team[j] == self.team[i] and j != i]
                                   for i in range(8)])
        # Fixed ray directions relative to the view (pitch rows x yaw columns).
        self._ray_yaw = np.linspace(FOV_H / 2, -FOV_H / 2, VISION_W)
        self._ray_pitch = np.linspace(FOV_V / 2, -FOV_V / 2, VISION_H)
        self.reset()

    # ---------- state ----------
    def reset(self):
        n = self.n_agents
        self.pos = np.zeros((n, 3))           # x, y, feet height
        self.yaw = np.zeros(n)
        self.pitch = np.zeros(n)
        self.crouch = np.zeros(n, dtype=bool)
        self.moving = np.zeros(n, dtype=bool)
        self.cooldown = np.zeros(n, dtype=int)
        self.down = np.zeros(n, dtype=int)    # steps until respawn; 0 = active
        self.score = np.zeros(n)
        self.radio = np.zeros((n, RADIO_DIGITS), dtype=int)
        self.last_shots = []                  # shots fired in the previous step
        self.was_hit_from = np.full(n, -1)    # sector a hit came from last step, -1 = none
        self.scored = np.zeros(n, dtype=bool)
        self.step_count = 0
        self.visible = np.zeros((n, n), dtype=bool)        # visible[i, j]: i can see j
        self.sight_cooldown = np.zeros((n, n), dtype=int)
        self.aim_score = np.zeros(n)
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

    def eyes(self):
        return np.column_stack([self.pos[:, 0], self.pos[:, 1],
                                self.pos[:, 2] + np.where(self.crouch, EYE_CROUCH, EYE_STAND)])

    def eye(self, i):
        return self.eyes()[i]

    def aims(self):
        cp = np.cos(self.pitch)
        return np.column_stack([cp * np.cos(self.yaw), cp * np.sin(self.yaw), np.sin(self.pitch)])

    def aim(self, i):
        return self.aims()[i]

    def hitboxes(self):
        """Axis-aligned hitboxes of all agents (down agents get an empty box far away)."""
        h = np.where(self.crouch, CROUCH_H, STAND_H)
        lo = np.column_stack([self.pos[:, 0] - RADIUS, self.pos[:, 1] - RADIUS, self.pos[:, 2]])
        hi = np.column_stack([self.pos[:, 0] + RADIUS, self.pos[:, 1] + RADIUS, self.pos[:, 2] + h])
        gone = self.down > 0
        lo[gone] = 1e6
        hi[gone] = 1e6
        return lo, hi

    def trace(self, origins, dirs, owners):
        """First thing each ray hits, ignoring the ray's owner.

        Returns (distance, agent id or -1 for world/nothing)."""
        lo, hi = self.hitboxes()
        return trace_rays(np.ascontiguousarray(origins, dtype=np.float64),
                          np.ascontiguousarray(dirs, dtype=np.float64),
                          np.asarray(owners, dtype=np.int64), self.arena.lo, self.arena.hi, lo, hi)

    # ---------- rules ----------
    def step(self, actions: list[Action]):
        """Advance one tick (0.1 s). Returns (observations, rewards, done, info)."""
        n = self.n_agents
        rewards = np.zeros(n)
        active = self.down == 0
        self.moving[:] = False

        # Movement, aiming and radio.
        for i in range(n):
            a = actions[i]
            self.radio[i] = np.clip(np.asarray(a.radio, dtype=int), 0, RADIO_BASE - 1)
            if not active[i]:
                continue
            self.crouch[i] = bool(a.crouch)
            self.yaw[i] = (self.yaw[i] + np.clip(a.turn, -1, 1) * TURN_SPEED * DT) % (2 * np.pi)
            self.pitch[i] = np.clip(self.pitch[i] + np.clip(a.pitch, -1, 1) * PITCH_SPEED * DT,
                                    -MAX_PITCH, MAX_PITCH)
            f, s = np.clip(a.forward, -1, 1), np.clip(a.strafe, -1, 1)
            if f or s:
                c, sn = np.cos(self.yaw[i]), np.sin(self.yaw[i])
                d = np.array([f * c + s * sn, f * sn - s * c])
                d = d / np.linalg.norm(d) * WALK_SPEED * DT * (0.5 if self.crouch[i] else 1.0)
                x, y, feet = self.pos[i]
                h = CROUCH_H if self.crouch[i] else STAND_H
                # Try the full move, then slide along each axis.
                r = (x, y, feet)
                for nx, ny in ((x + d[0], y + d[1]), (x + d[0], y), (x, y + d[1])):
                    r = self.arena.try_move(x, y, feet, nx, ny, RADIUS, h)
                    if (r[0], r[1]) != (x, y):
                        break
                self.pos[i] = r
                self.moving[i] = (r[0], r[1]) != (x, y)

        # Firing: all shots this tick resolve together, against positions after movement.
        shooters = np.array([i for i in range(n) if active[i] and actions[i].fire and self.cooldown[i] == 0],
                            dtype=int)
        shots, tagged = [], {}
        self.was_hit_from[:] = -1
        self.scored[:] = False
        if len(shooters):
            eyes, aims = self.eyes()[shooters], self.aims()[shooters]
            dist, who = self.trace(eyes, aims, shooters)
            for i, d, j in zip(shooters, dist, who):
                i, j = int(i), int(j)
                self.cooldown[i] = COOLDOWN_STEPS
                end = self.eye(i) + self.aim(i) * min(float(d), 60.0)
                shots.append({"shooter": i, "target": j, "distance": float(d), "end": end.round(2).tolist()})
                if j >= 0 and self.team[j] != self.team[i] and j not in tagged:
                    tagged[j] = i
                    rewards[i] += HIT_REWARD
                    rewards[j] += TAGGED_REWARD
                    self.scored[i] = True
                    dx, dy = self.pos[i, :2] - self.pos[j, :2]
                    self.was_hit_from[j] = _sector(np.arctan2(dy, dx) - self.yaw[j])
        for j in tagged:
            self.down[j] = RESPAWN_STEPS
        self.last_shots = shots

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
        prev_visible = self.visible.copy()
        obs = self.observe_all()
        sightings = []
        if self.aim_rewards:
            enemy = self.team[:, None] != self.team[None, :]
            entered = self.visible & ~prev_visible & enemy & (self.sight_cooldown == 0)
            aim = entered.sum(1) * SIGHT_REWARD + entered.sum(0) * SEEN_PENALTY
            rewards = rewards + aim
            self.aim_score += aim
            self.sight_cooldown[entered] = SIGHT_RETRIGGER_STEPS
            sightings = [{"viewer": int(i), "seen": int(j)} for i, j in zip(*np.nonzero(entered))]
        self.sight_cooldown = np.maximum(self.sight_cooldown - 1, 0)
        return obs, rewards, done, {"shots": shots, "tagged": tagged, "sightings": sightings,
                                    "team_scores": self.team_scores()}

    def team_scores(self):
        return [float(self.score[self.team == t].sum()) for t in (GREEN, PINK)]

    def winner(self):
        g, p = self.team_scores()
        return "draw" if g == p else TEAM_NAMES[int(p > g)]

    # ---------- perception ----------
    def observe_all(self):
        """What each agent can perceive, batched for all 8 agents.

        vision  (8, VISION_H, VISION_W, 4): per ray [distance / MAX_VIEW, hit code / 3,
                 how directly the seen player aims at me (0..1), seen player's gun ready].
                 Walls block sight; nothing outside the 90 x 60 degree view is visible.
        body    (8, BODY_SIZE): cooldown, down timer, crouching, height, pitch, time left, moving.
        hearing (8, 16): footsteps and gunshots within 15 m, loudness per 45-degree sector.
        events  (8, 10): I was tagged, I tagged someone, direction the tag came from.
        radio   (8, 120): teammates' last four-digit codes, one-hot.
        """
        n = self.n_agents
        P, Y = np.meshgrid(self._ray_pitch, self._ray_yaw, indexing="ij")
        P = P.ravel()[None] + self.pitch[:, None]
        Y = Y.ravel()[None] + self.yaw[:, None]
        dirs = np.stack([np.cos(P) * np.cos(Y), np.cos(P) * np.sin(Y), np.sin(P)], -1).reshape(-1, 3)
        rays = VISION_W * VISION_H
        owners = np.repeat(np.arange(n), rays)
        eyes = self.eyes()
        dist, who = self.trace(eyes[owners], dirs, owners)

        seen = who >= 0
        whoc = np.maximum(who, 0)
        same = self.team[whoc] == self.team[owners]
        code = np.where(seen, np.where(same, SEE_TEAMMATE, SEE_OPPONENT),
                        np.where(np.isfinite(dist) & (dist <= MAX_VIEW), SEE_WORLD, SEE_NOTHING))
        to_me = eyes[owners] - eyes[whoc]
        to_me /= np.linalg.norm(to_me, axis=1, keepdims=True) + 1e-9
        aimed = np.where(seen, np.clip((self.aims()[whoc] * to_me).sum(1), 0, 1), 0.0)
        ready = np.where(seen, (self.cooldown[whoc] == 0).astype(float), 0.0)
        d = np.minimum(np.nan_to_num(dist, posinf=MAX_VIEW), MAX_VIEW) / MAX_VIEW
        vision = np.stack([d, code / 3.0, aimed, ready], -1).reshape(n, VISION_H, VISION_W, VISION_CHANNELS)

        # Rays are ~4 degrees apart, so a distant player can slip between them even
        # though a person would clearly see them. For every other player inside the
        # view cone with a clear line of sight to their chest, mark the nearest cell.
        self.visible = np.zeros((n, n), dtype=bool)
        ii, jj = np.nonzero(~np.eye(n, dtype=bool) & (self.down[None, :] == 0) & (self.down[:, None] == 0))
        chest = self.pos[jj] + np.array([0, 0, 1.0]) * np.where(self.crouch[jj], 0.7, 1.1)[:, None]
        vec = chest - eyes[ii]
        rng_ = np.linalg.norm(vec, axis=1)
        bearing = (np.arctan2(vec[:, 1], vec[:, 0]) - self.yaw[ii] + np.pi) % (2 * np.pi) - np.pi
        elev = np.arcsin(np.clip(vec[:, 2] / (rng_ + 1e-9), -1, 1)) - self.pitch[ii]
        inview = (np.abs(bearing) <= FOV_H / 2) & (np.abs(elev) <= FOV_V / 2) & (rng_ <= MAX_VIEW)
        if inview.any():
            ii, jj, vec, rng_, bearing, elev = (x[inview] for x in (ii, jj, vec, rng_, bearing, elev))
            t, hit = self.trace(eyes[ii], vec / rng_[:, None], ii)
            vis = hit == jj
            self.visible[ii[vis], jj[vis]] = True
            for i, j, r, b, e in zip(ii[vis], jj[vis], rng_[vis], bearing[vis], elev[vis]):
                col = int(np.argmin(np.abs(self._ray_yaw - b)))
                row = int(np.argmin(np.abs(self._ray_pitch - e)))
                cell = vision[i, row, col]
                if cell[1] * 3 >= SEE_TEAMMATE and cell[0] < r / MAX_VIEW:
                    continue  # a closer player already fills this cell
                to_me = eyes[i] - eyes[j]
                to_me /= np.linalg.norm(to_me) + 1e-9
                cell[:] = (r / MAX_VIEW,
                           (SEE_TEAMMATE if self.team[i] == self.team[j] else SEE_OPPONENT) / 3.0,
                           max(0.0, float(self.aims()[j] @ to_me)), float(self.cooldown[j] == 0))

        body = np.column_stack([
            self.cooldown / COOLDOWN_STEPS, self.down / RESPAWN_STEPS, self.crouch.astype(float),
            self.pos[:, 2] / 1.5, self.pitch / MAX_PITCH,
            np.full(n, 1.0 - self.step_count / MATCH_STEPS), self.moving.astype(float)])

        # Hearing: other players' footsteps (if moving and not crouched) and gunshots.
        hearing = np.zeros((n, HEARING_SIZE))
        shot_src = [s["shooter"] for s in self.last_shots]
        for i in range(n):
            for j in range(n):
                if j == i or self.down[j] > 0:
                    continue
                delta = self.pos[j, :2] - self.pos[i, :2]
                r = np.hypot(*delta)
                if r > HEARING_RANGE:
                    continue
                sec = _sector(np.arctan2(delta[1], delta[0]) - self.yaw[i])
                loud = 1.0 - r / HEARING_RANGE
                if self.moving[j] and not self.crouch[j]:
                    hearing[i, sec] = max(hearing[i, sec], loud)
                if j in shot_src:
                    hearing[i, SECTORS + sec] = max(hearing[i, SECTORS + sec], loud)

        events = np.zeros((n, EVENT_SIZE))
        events[:, 0] = self.was_hit_from >= 0
        events[:, 1] = self.scored
        hit = self.was_hit_from >= 0
        events[np.flatnonzero(hit), 2 + self.was_hit_from[hit]] = 1.0

        radio = np.zeros((n, 3, RADIO_DIGITS, RADIO_BASE))
        mates = self.radio[self.teammates]                      # (8, 3, 4)
        np.put_along_axis(radio, mates[..., None], 1.0, axis=3)
        return {"vision": vision.astype(np.float32), "body": body.astype(np.float32),
                "hearing": hearing.astype(np.float32), "events": events.astype(np.float32),
                "radio": radio.reshape(n, -1).astype(np.float32)}

    def observe(self, i):
        """Observation of a single agent (convenience wrapper)."""
        return {k: v[i] for k, v in self.observe_all().items()}
