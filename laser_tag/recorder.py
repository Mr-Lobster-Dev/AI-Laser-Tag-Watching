"""Record a full match (positions, shots, radio, and network activations) to a replay file.

The file format is documented in docs/REPLAY_FORMAT.md. Replays are gzipped JSON;
large per-step arrays are stored as base64-encoded uint8 to keep files small.
"""
from __future__ import annotations

import base64
import gzip
import json
import os

import numpy as np

from .arena import WIDTH, DEPTH, WALL_H
from .game import TEAM_NAMES, TEAM_SHAPES, DT, MATCH_STEPS, SEE_OPPONENT
from .model import HEADS, ENC, HID

FORMAT_VERSION = 1
KIND_NAMES = ["floor", "wall", "cover", "platform", "stair"]


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode("ascii")


def _q(x, lo, hi):
    """Quantize floats in [lo, hi] to uint8."""
    return np.clip(np.rint((np.asarray(x, dtype=np.float32) - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)


class MatchRecorder:
    def __init__(self, game, match_number, trained_matches):
        self.g = game
        self.match = match_number
        self.trained = trained_matches
        self.pos, self.yaw, self.pitch, self.crouch, self.down, self.ready = [], [], [], [], [], []
        self.radio, self.scores, self.shots = [], [], []
        self.vision, self.enc_v, self.enc_o, self.hidden, self.probs, self.value = [], [], [], [], [], []

    def record(self, obs, info, net):
        """Call once per step, after game.step(). obs: this agent-batch's obs dict for the
        new state. net: dict with numpy arrays enc_v (8,64), enc_o (8,64), h (8,128),
        probs (8,58), value (8,) from the forward pass that chose this step's actions."""
        g = self.g
        step = g.step_count
        self.pos.append(g.pos.round(2))
        self.yaw.append(g.yaw.round(3))
        self.pitch.append(g.pitch.round(3))
        self.crouch.append(g.crouch.astype(int))
        self.down.append((g.down > 0).astype(int))
        self.ready.append((g.cooldown == 0).astype(int))
        self.radio.append(["".join(map(str, r)) for r in g.radio])
        self.scores.append(g.team_scores())
        for s in info["shots"]:
            j = s["target"]
            tag = j >= 0 and g.team[j] != g.team[s["shooter"]] and info["tagged"].get(j) == s["shooter"]
            self.shots.append({"step": step, "shooter": s["shooter"], "target": j, "tag": bool(tag),
                               "end": s["end"]})
        # Vision: one byte per ray = (distance quantized to 6 bits) << 2 | hit code.
        v = obs["vision"]
        code = np.rint(v[..., 1] * 3).astype(np.uint8)
        dist = np.minimum(np.rint(v[..., 0] * 63), 63).astype(np.uint8)
        self.vision.append((dist << 2) | code)
        self.enc_v.append(_q(net["enc_v"], 0, 4))
        self.enc_o.append(_q(net["enc_o"], 0, 4))
        self.hidden.append(_q(net["h"], -1, 1))
        self.probs.append(_q(net["probs"], 0, 1))
        self.value.append(np.round(net["value"], 2))

    def save(self, path, policy_weights=None):
        g = self.g
        steps = len(self.pos)
        boxes = [[*map(float, lo), *map(float, hi), KIND_NAMES[k]] for lo, hi, k in g.arena.boxes if k != 0]
        doc = {
            "version": FORMAT_VERSION,
            "match": self.match,
            "trained_matches": self.trained,
            "dt": DT,
            "steps": steps,
            "arena": {"width": WIDTH, "depth": DEPTH, "wall_height": WALL_H, "boxes": boxes},
            "agents": [{"id": i, "team": TEAM_NAMES[g.team[i]], "shape": TEAM_SHAPES[g.team[i]]}
                       for i in range(g.n_agents)],
            "final_scores": {"green": g.team_scores()[0], "pink": g.team_scores()[1],
                             "per_agent": g.score.tolist(), "winner": g.winner()},
            "frames": {
                "pos": np.array(self.pos).ravel().tolist(),
                "yaw": np.array(self.yaw).ravel().tolist(),
                "pitch": np.array(self.pitch).ravel().tolist(),
                "crouch": np.array(self.crouch).ravel().tolist(),
                "down": np.array(self.down).ravel().tolist(),
                "gun_ready": np.array(self.ready).ravel().tolist(),
                "radio": [c for row in self.radio for c in row],
                "team_scores": np.array(self.scores).ravel().tolist(),
            },
            "shots": self.shots,
            "network": {
                "heads": [{"name": n, "size": s} for n, s in HEADS],
                "vision": {"rows": 8, "cols": 24, "data": _b64(np.array(self.vision))},
                "enc_vision": {"size": ENC, "range": [0, 4], "data": _b64(np.array(self.enc_v))},
                "enc_other": {"size": ENC, "range": [0, 4], "data": _b64(np.array(self.enc_o))},
                "memory": {"size": HID, "range": [-1, 1], "data": _b64(np.array(self.hidden))},
                "action_probs": {"size": sum(s for _, s in HEADS), "range": [0, 1],
                                 "data": _b64(np.array(self.probs))},
                "value": np.array(self.value).ravel().tolist(),
            },
        }
        if policy_weights is not None:
            # memory -> action logits weights per agent, (8, 128, 58), quantized to int8 as uint8+128.
            w = np.asarray(policy_weights, dtype=np.float32)
            scale = float(np.abs(w).max()) or 1.0
            doc["network"]["policy_weights"] = {"shape": list(w.shape), "scale": scale,
                                                "data": _b64(np.rint(w / scale * 127) + 128)}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt", compresslevel=6) as f:
            json.dump(doc, f, separators=(",", ":"))
        os.replace(tmp, path)
        return path
