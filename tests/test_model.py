import torch

from laser_tag.game import LaserTagGame
from laser_tag.model import SquadNet, obs_to_tensors, sample, evaluate, to_action, N_LOGITS
from laser_tag.train import should_record


def test_forward_and_actions():
    g = LaserTagGame(seed=1)
    net = SquadNet()
    v, o = obs_to_tensors([g.reset(), g.reset()])
    logits, value, h, _ = net.step(v, o, net.initial_state(2))
    assert logits.shape == (8, 2, N_LOGITS) and value.shape == (8, 2) and h.shape == (8, 2, 128)
    acts, logp = sample(logits)
    lp2, ent = evaluate(logits, acts)
    assert torch.allclose(logp, lp2)
    a = to_action(acts[0, 0])
    assert len(a.radio) == 4 and all(0 <= d <= 9 for d in a.radio)


def test_agents_have_separate_weights():
    net = SquadNet()
    w = net.vision.weight
    assert not torch.equal(w[0], w[1])


def test_records_first_and_every_100th():
    assert [m for m in range(1, 401) if should_record(m)] == [1, 100, 200, 300, 400]


def test_motion_is_60fps_and_does_not_glide_on_respawn():
    import numpy as np
    from laser_tag.recorder import motion_60fps
    S = 3
    pos = np.zeros((S, 8, 3)); pos[1, 0, 0] = 0.3; pos[2, 0, 0] = 20.0   # agent 0 respawns at step 2
    yaw = np.zeros((S, 8)); yaw[0, 1] = 6.2; yaw[1, 1] = 0.1              # wraps around
    p, y, _ = motion_60fps(pos, yaw, np.zeros((S, 8)), np.zeros((S, 8), dtype=int))
    assert p.shape[0] == (S - 1) * 6 + 1
    assert abs(p[3, 0, 0] - 0.15) < 1e-9                  # halfway between steps 0 and 1
    assert (p[6:12, 0, 0] == 0.3).all() and p[12, 0, 0] == 20.0   # no glide on respawn
    assert all(min(v, 2 * np.pi - v) < 0.2 for v in y[:7, 1])       # short way round
