"""Eight small recurrent policy networks, one per agent, evaluated together.

Every agent has its own weights (nothing is shared between agents), but the
weights are stored as stacked tensors so all 8 networks run in one batched call.

Per agent (about 165k parameters):
    vision (8 x 24 x 4 = 768)  -> 64   "what I see"
    body + hearing + events + teammates' radio codes (153) -> 64
    concat (128) -> GRU memory (128)
    memory -> action logits (58) and value estimate (1)
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .game import (Action, VISION_W, VISION_H, VISION_CHANNELS, BODY_SIZE, HEARING_SIZE,
                   EVENT_SIZE, RADIO_IN_SIZE, RADIO_DIGITS, RADIO_BASE)

N_AGENTS = 8
VISION_IN = VISION_W * VISION_H * VISION_CHANNELS
OTHER_IN = BODY_SIZE + HEARING_SIZE + EVENT_SIZE + RADIO_IN_SIZE
ENC, HID = 64, 128

# Action heads (multi-discrete).
FORWARD = (-1, 0, 1)
STRAFE = (-1, 0, 1)
TURN = (-1.0, -0.4, 0.0, 0.4, 1.0)
PITCH = (-0.5, 0.0, 0.5)
HEADS = [("forward", 3), ("strafe", 3), ("turn", 5), ("pitch", 3), ("crouch", 2), ("fire", 2)] + \
        [(f"radio{k}", RADIO_BASE) for k in range(RADIO_DIGITS)]
HEAD_SIZES = [s for _, s in HEADS]
N_LOGITS = sum(HEAD_SIZES)


class GroupLinear(nn.Module):
    """G independent linear layers applied to x of shape (G, B, in)."""

    def __init__(self, groups, n_in, n_out):
        super().__init__()
        bound = 1 / math.sqrt(n_in)
        self.weight = nn.Parameter(torch.empty(groups, n_in, n_out).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(groups, 1, n_out))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class SquadNet(nn.Module):
    def __init__(self, groups=N_AGENTS):
        super().__init__()
        self.groups = groups
        self.vision = GroupLinear(groups, VISION_IN, ENC)
        self.other = GroupLinear(groups, OTHER_IN, ENC)
        self.gru_x = GroupLinear(groups, 2 * ENC, 3 * HID)
        self.gru_h = GroupLinear(groups, HID, 3 * HID)
        self.policy = GroupLinear(groups, HID, N_LOGITS)
        self.value = GroupLinear(groups, HID, 1)
        with torch.no_grad():
            self.policy.weight.mul_(0.01)
            # Start out sensible: mostly keep pitch level, and fire whenever ready
            # (a shot costs nothing but the cooldown).
            b = self.policy.bias.view(groups, N_LOGITS)
            off = np.cumsum([0] + HEAD_SIZES)
            pitch = off[3]
            b[:, pitch + 1] = 2.0
            fire = off[5]
            b[:, fire + 1] = 1.0

    def initial_state(self, batch):
        return torch.zeros(self.groups, batch, HID)

    def step(self, vision, other, h):
        """One time step. vision (G,B,768), other (G,B,153), h (G,B,128).
        Returns logits, value, new h, and the intermediate activations."""
        ev = F.relu(self.vision(vision))
        eo = F.relu(self.other(other))
        x = torch.cat([ev, eo], -1)
        gx, gh = self.gru_x(x), self.gru_h(h)
        rx, zx, nx = gx.chunk(3, -1)
        rh, zh, nh = gh.chunk(3, -1)
        r = torch.sigmoid(rx + rh)
        z = torch.sigmoid(zx + zh)
        n = torch.tanh(nx + r * nh)
        h = (1 - z) * n + z * h
        return self.policy(h), self.value(h).squeeze(-1), h, (ev, eo)


def obs_to_tensors(obs_list):
    """List of per-env observation dicts -> (vision, other) tensors of shape (8, E, ...)."""
    vision = np.stack([o["vision"].reshape(N_AGENTS, -1) for o in obs_list], 1)
    other = np.stack([np.concatenate([o["body"], o["hearing"], o["events"], o["radio"]], -1)
                      for o in obs_list], 1)
    return torch.from_numpy(vision), torch.from_numpy(other)


def split_logits(logits):
    return torch.split(logits, HEAD_SIZES, dim=-1)


def sample(logits, greedy=False):
    """Sample every head. Returns (actions (..., n_heads) long, total log prob)."""
    acts, logp = [], 0
    for lg in split_logits(logits):
        dist = torch.distributions.Categorical(logits=lg)
        a = lg.argmax(-1) if greedy else dist.sample()
        acts.append(a)
        logp = logp + dist.log_prob(a)
    return torch.stack(acts, -1), logp


def evaluate(logits, acts):
    """Log prob and entropy of given actions."""
    logp, ent = 0, 0
    for k, lg in enumerate(split_logits(logits)):
        dist = torch.distributions.Categorical(logits=lg)
        logp = logp + dist.log_prob(acts[..., k])
        ent = ent + dist.entropy()
    return logp, ent


def to_action(a):
    """Head indices (n_heads,) -> game Action."""
    a = [int(v) for v in a]
    return Action(forward=FORWARD[a[0]], strafe=STRAFE[a[1]], turn=TURN[a[2]], pitch=PITCH[a[3]],
                  crouch=bool(a[4]), fire=bool(a[5]), radio=tuple(a[6:6 + RADIO_DIGITS]))
