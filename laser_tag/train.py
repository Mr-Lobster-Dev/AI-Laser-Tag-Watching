"""PPO self-play training for the 8 laser tag agents.

Each agent has its own network and learns from its own reward (+5 tag, -3 tagged).
Several matches run side by side; after each batch the networks are updated with
PPO (recurrent, truncated backprop through time in chunks).

Match 1, every 100th match, and the first match in which anyone gets tagged
are recorded as replays.

Usage:
    python -m laser_tag.train --out /mnt/project-files/laser-tag   # resumes automatically
    python -m laser_tag.train --matches 8 --envs 4                   # quick smoke run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from .game import LaserTagGame, MATCH_STEPS
from .model import (SquadNet, N_AGENTS, obs_to_tensors, sample, evaluate, to_action,
                    split_logits)
from .recorder import MatchRecorder


def should_record(match_no, every=100):
    return match_no == 1 or match_no % every == 0


class Trainer:
    def __init__(self, out, envs=4, lr=3e-4, gamma=0.995, lam=0.95, clip=0.2, epochs=3,
                 chunk=40, minibatches=4, ent_coef=0.01, vf_coef=0.5, reward_scale=0.2, seed=0):
        self.out = out
        self.E = envs
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.chunk, self.minibatches = epochs, chunk, minibatches
        self.ent_coef, self.vf_coef, self.reward_scale = ent_coef, vf_coef, reward_scale
        torch.manual_seed(seed)
        self.net = SquadNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.matches_done = 0
        self.seed = seed
        os.makedirs(os.path.join(out, "replays"), exist_ok=True)
        os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
        self.ckpt = os.path.join(out, "checkpoints", "latest.pt")
        self.log_path = os.path.join(out, "train_log.csv")
        if os.path.exists(self.ckpt):
            state = torch.load(self.ckpt, weights_only=False)
            self.net.load_state_dict(state["net"])
            self.opt.load_state_dict(state["opt"])
            self.matches_done = state["matches_done"]
            print(f"Resumed from {self.ckpt} after {self.matches_done} matches")

    # ---------- rollout ----------
    @torch.no_grad()
    def collect(self):
        E, G, T = self.E, N_AGENTS, MATCH_STEPS
        first = self.matches_done + 1
        games = [LaserTagGame(seed=self.seed * 1_000_003 + first + e) for e in range(E)]
        numbers = [first + e for e in range(E)]
        first_hit_marker = os.path.join(self.out, "replays", "first_hit.json")
        hunting = not os.path.exists(first_hit_marker)   # still looking for the first tag
        recorders = {e: MatchRecorder(games[e], numbers[e], self.matches_done)
                     for e in range(E) if hunting or should_record(numbers[e])}
        env_tags = [0] * E
        obs = [g.reset() for g in games]
        h = self.net.initial_state(E)

        buf_v = torch.zeros(T, G, E, 768, dtype=torch.float16)
        buf_o = torch.zeros(T, G, E, obs_to_tensors(obs)[1].shape[-1])
        buf_h = torch.zeros(T, G, E, h.shape[-1])
        buf_a = torch.zeros(T, G, E, 10, dtype=torch.long)
        buf_lp = torch.zeros(T, G, E)
        buf_val = torch.zeros(T, G, E)
        buf_r = torch.zeros(T, G, E)
        shots = tags = 0

        for t in range(T):
            v, o = obs_to_tensors(obs)
            buf_v[t], buf_o[t], buf_h[t] = v.half(), o, h
            logits, value, h, (ev, eo) = self.net.step(v, o, h)
            acts, logp = sample(logits)
            buf_a[t], buf_lp[t], buf_val[t] = acts, logp, value
            acts_np = acts.numpy()
            for e, g in enumerate(games):
                ob, rew, _, info = g.step([to_action(acts_np[i, e]) for i in range(G)])
                obs[e] = ob
                buf_r[t, :, e] = torch.from_numpy(rew).float()
                shots += len(info["shots"])
                tags += len(info["tagged"])
                env_tags[e] += len(info["tagged"])
                if e in recorders:
                    probs = torch.cat([torch.softmax(lg[:, e], -1) for lg in split_logits(logits)], -1)
                    recorders[e].record(ob, info, {
                        "enc_v": ev[:, e].numpy(), "enc_o": eo[:, e].numpy(), "h": h[:, e].numpy(),
                        "probs": probs.numpy(), "value": value[:, e].numpy() / self.reward_scale})

        w = self.net.policy.weight.detach().numpy()
        first_hit = None
        if hunting:
            hits = [e for e in range(E) if env_tags[e] > 0]
            if hits:
                first_hit = hits[0]
        for e, rec in recorders.items():
            if not (should_record(numbers[e]) or e == first_hit):
                continue
            path = os.path.join(self.out, "replays", f"match_{numbers[e]:06d}.json.gz")
            rec.save(path, policy_weights=w)
            print(f"  recorded match {numbers[e]} -> {path}")
        if first_hit is not None:
            with open(first_hit_marker, "w") as f:
                json.dump({"match": numbers[first_hit], "file": f"match_{numbers[first_hit]:06d}.json.gz",
                           "tags": env_tags[first_hit]}, f)
            print(f"  first match with a tag: {numbers[first_hit]}")

        scores = np.array([g.team_scores() for g in games])
        self.matches_done += E
        stats = {"matches": self.matches_done, "green_mean": scores[:, 0].mean(),
                 "pink_mean": scores[:, 1].mean(), "tags_per_match": tags / E,
                 "shots_per_match": shots / E}
        return (buf_v, buf_o, buf_h, buf_a, buf_lp, buf_val, buf_r), stats

    # ---------- PPO update ----------
    def update(self, batch):
        v, o, hs, a, old_lp, val, r = batch
        T, G, E = r.shape
        r = r * self.reward_scale
        adv = torch.zeros_like(r)
        last = torch.zeros(G, E)
        for t in reversed(range(T)):
            next_v = val[t + 1] if t + 1 < T else torch.zeros(G, E)
            delta = r[t] + self.gamma * next_v - val[t]
            last = delta + self.gamma * self.lam * last
            adv[t] = last
        ret = adv + val

        # Reshape (T, G, E, ...) -> chunks (G, C, L, ...), C = E * T / L.
        L = self.chunk
        n = T // L

        def chunks(x):
            x = x[: n * L].reshape(n, L, G, E, *x.shape[3:])           # (n, L, G, E, ...)
            x = x.permute(2, 3, 0, 1, *range(4, x.dim()))              # (G, E, n, L, ...)
            return x.reshape(G, E * n, L, *x.shape[4:])

        v, o, a, old_lp, adv, ret = map(chunks, (v, o, a, old_lp, adv, ret))
        h0 = chunks(hs)[:, :, 0]                                       # (G, C, H)
        adv = (adv - adv.mean(dim=(1, 2), keepdim=True)) / (adv.std(dim=(1, 2), keepdim=True) + 1e-8)
        C = v.shape[1]
        stats = []
        for _ in range(self.epochs):
            perm = torch.randperm(C)
            for idx in perm.chunk(self.minibatches):
                h = h0[:, idx]
                logits_seq, values_seq = [], []
                for t in range(L):
                    lg, vl, h, _ = self.net.step(v[:, idx, t].float(), o[:, idx, t], h)
                    logits_seq.append(lg)
                    values_seq.append(vl)
                logits = torch.stack(logits_seq, 2)
                values = torch.stack(values_seq, 2)
                lp, ent = evaluate(logits, a[:, idx])
                ratio = torch.exp(lp - old_lp[:, idx])
                ad = adv[:, idx]
                pg = -torch.min(ratio * ad, ratio.clamp(1 - self.clip, 1 + self.clip) * ad).mean()
                vf = ((values - ret[:, idx]) ** 2).mean()
                loss = pg + self.vf_coef * vf - self.ent_coef * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                stats.append((pg.item(), vf.item(), ent.mean().item()))
        pg, vf, ent = np.mean(stats, 0)
        return {"policy_loss": pg, "value_loss": vf, "entropy": ent}

    def save(self):
        tmp = self.ckpt + ".tmp"
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict(),
                    "matches_done": self.matches_done}, tmp)
        os.replace(tmp, self.ckpt)
        if self.matches_done % 500 < self.E:
            torch.save({"net": self.net.state_dict(), "matches_done": self.matches_done},
                       os.path.join(self.out, "checkpoints", f"match_{self.matches_done:06d}.pt"))

    def log(self, row):
        new = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def run(self, total_matches):
        while self.matches_done < total_matches:
            t0 = time.time()
            batch, stats = self.collect()
            t1 = time.time()
            stats.update(self.update(batch))
            stats["collect_s"], stats["update_s"] = round(t1 - t0, 1), round(time.time() - t1, 1)
            self.save()
            self.log({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()})
            print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()),
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/local")
    ap.add_argument("--matches", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    Trainer(args.out, envs=args.envs, seed=args.seed).run(args.matches)


if __name__ == "__main__":
    main()
