import numpy as np

from laser_tag.game import (LaserTagGame, Action, COOLDOWN_STEPS, RESPAWN_STEPS, MATCH_STEPS,
                            SEE_OPPONENT, SEE_TEAMMATE)

IDLE = Action()


def place(g, i, x, y, yaw, feet=0.0):
    g.pos[i] = (x, y, feet)
    g.yaw[i] = yaw
    g.pitch[i] = 0.0


def clear_field(g, keep):
    """Park everyone except `keep` in a far corner, out of the way."""
    for k in range(8):
        if k not in keep:
            place(g, k, 0.5 + 0.7 * (k % 4), 19.5 if k < 4 else 0.5, 0.0)


def fire(i):
    acts = [IDLE] * 8
    acts = list(acts)
    acts[i] = Action(fire=True)
    return acts


def test_hit_opponent_scores_plus5_minus3():
    g = LaserTagGame(seed=0)
    clear_field(g, {0, 4})
    place(g, 0, 2.0, 1.0, 0.0)          # open lane along y=1
    place(g, 4, 25.0, 1.0, np.pi)
    _, r, _, info = g.step(fire(0))
    assert info["shots"][0]["target"] == 4
    assert r[0] == 5 and r[4] == -3
    assert g.team_scores() == [5.0, -3.0]
    assert g.down[4] == RESPAWN_STEPS


def test_friendly_fire_does_nothing():
    g = LaserTagGame(seed=0)
    clear_field(g, {0, 1})
    place(g, 0, 2.0, 1.0, 0.0)
    place(g, 1, 6.0, 1.0, 0.0)
    _, r, _, _ = g.step(fire(0))
    assert r.sum() == 0 and g.down[1] == 0


def test_walls_block_lasers_and_no_bounce():
    g = LaserTagGame(seed=0)
    clear_field(g, {0, 4})
    place(g, 0, 8.0, 6.5, 0.0)          # wall at x=10..10.3 spans y 4..9
    place(g, 4, 12.0, 6.5, np.pi)
    _, r, _, info = g.step(fire(0))
    assert info["shots"][0]["target"] == -1 and r.sum() == 0


def test_cooldown_five_seconds():
    g = LaserTagGame(seed=0)
    g.step(fire(0))
    for _ in range(COOLDOWN_STEPS - 1):
        _, _, _, info = g.step(fire(0))
        assert not info["shots"]
    _, _, _, info = g.step(fire(0))
    assert len(info["shots"]) == 1


def test_respawn_after_three_seconds_on_own_end():
    g = LaserTagGame(seed=0)
    clear_field(g, {0, 4})
    place(g, 0, 2.0, 1.0, 0.0)
    place(g, 4, 25.0, 1.0, np.pi)
    g.step(fire(0))
    for _ in range(RESPAWN_STEPS):
        g.step([IDLE] * 8)
    assert g.down[4] == 0 and g.pos[4, 0] > 26.5


def test_cannot_walk_through_walls_but_can_climb_stairs():
    g = LaserTagGame(seed=0)
    clear_field(g, {0})
    place(g, 0, 9.5, 6.5, 0.0)
    for _ in range(20):
        g.step([Action(forward=1)] + [IDLE] * 7)
    assert g.pos[0, 0] < 10.0
    # Stairs run +x from x=8..10 at y 14..16 down from the 1.5 m platform at x 5..8.
    place(g, 0, 10.6, 15.0, np.pi)
    for _ in range(15):
        g.step([Action(forward=1)] + [IDLE] * 7)
    assert g.pos[0, 2] == 1.5


def test_vision_sees_opponent_not_through_walls():
    g = LaserTagGame(seed=0)
    clear_field(g, {0, 1, 4})
    place(g, 0, 2.0, 1.0, 0.0)
    place(g, 4, 20.0, 1.0, np.pi)
    codes = np.rint(g.observe(0)["vision"][..., 1] * 3)
    assert (codes == SEE_OPPONENT).any()
    place(g, 0, 8.0, 6.5, 0.0)
    place(g, 4, 12.0, 6.5, np.pi)
    codes = np.rint(g.observe(0)["vision"][..., 1] * 3)
    assert not (codes == SEE_OPPONENT).any()


def test_match_lasts_three_minutes():
    g = LaserTagGame(seed=0)
    g.observe_all = lambda: None  # skip vision for speed
    done = False
    for k in range(MATCH_STEPS):
        _, _, done, _ = g.step([IDLE] * 8)
        assert done == (k == MATCH_STEPS - 1)


def test_radio_reaches_teammates_only():
    g = LaserTagGame(seed=0)
    acts = [IDLE] * 8
    acts = list(acts)
    acts[0] = Action(radio=(1, 2, 3, 9))
    obs, _, _, _ = g.step(acts)
    r = obs["radio"].reshape(8, 3, 4, 10)
    # Agent 1's first teammate slot is agent 0.
    assert r[1, 0].argmax(axis=1).tolist() == [1, 2, 3, 9]
    # Pink agents never receive green codes: their radio inputs show only pink codes (all 0).
    assert (r[4:, :, :, 0] == 1).all()


def test_sees_aimed_gun():
    g = LaserTagGame(seed=0)
    clear_field(g, {0, 4})
    place(g, 0, 2.0, 1.0, 0.0)
    place(g, 4, 20.0, 1.0, np.pi)      # pink aims straight back at green
    v = g.observe(0)["vision"]
    opp = np.rint(v[..., 1] * 3) == SEE_OPPONENT
    assert v[..., 2][opp].max() > 0.95 and v[..., 3][opp].min() == 1.0


def test_hearing_and_hit_direction():
    g = LaserTagGame(seed=0)
    clear_field(g, {0, 4})
    place(g, 0, 2.0, 1.0, 0.0)
    place(g, 4, 12.0, 1.0, np.pi)
    obs, _, _, _ = g.step(fire(0))
    assert obs["events"][4, 0] == 1 and obs["events"][0, 1] == 1
    assert obs["events"][4, 2 + 0] == 1           # shot came from straight ahead


def test_aim_rewards_plus1_seeing_minus1_seen_with_retrigger_delay():
    g = LaserTagGame(seed=0, aim_rewards=True)
    clear_field(g, {0, 4})
    place(g, 0, 2.0, 1.0, np.pi)        # green looks away from pink
    place(g, 4, 20.0, 1.0, 0.0)         # pink looks away from green
    g.step([IDLE] * 8)
    g.yaw[0] = 0.0                      # green turns to face pink
    _, r, _, info = g.step([IDLE] * 8)
    assert info["sightings"] == [{"viewer": 0, "seen": 4}]
    assert r[0] == 1 and r[4] == -1
    assert g.team_scores() == [0.0, 0.0]          # scoreboard is tag points only
    g.yaw[0] = np.pi; g.step([IDLE] * 8)
    g.yaw[0] = 0.0
    _, r, _, _ = g.step([IDLE] * 8)
    assert r[0] == 0                    # looking away and back within 3 s doesn't count again


def test_no_aim_rewards_by_default():
    g = LaserTagGame(seed=0)
    clear_field(g, {0, 4})
    place(g, 0, 2.0, 1.0, np.pi)
    place(g, 4, 20.0, 1.0, 0.0)
    g.step([IDLE] * 8)
    g.yaw[0] = 0.0
    _, r, _, _ = g.step([IDLE] * 8)
    assert r.sum() == 0


def _duel_setup(hints=True, target_yaw=np.pi):
    g = LaserTagGame(seed=0, duel_hints=hints)
    clear_field(g, {0, 4})
    place(g, 0, 2.0, 1.0, 0.0)
    place(g, 4, 20.0, 1.0, target_yaw)
    g.step([IDLE] * 8)                  # let both register what they can see
    return g


def test_faceoff_hint_plus10_without_changing_score():
    g = _duel_setup(target_yaw=np.pi)   # pink aims straight at green
    _, r, _, info = g.step(fire(0))
    assert info["duels"][0]["kind"] == "faceoff"
    assert r[0] == 5 + 10 and r[4] == -3
    assert g.team_scores() == [5.0, -3.0]


def test_duel_hint_plus7_when_seen_but_not_aimed():
    g = _duel_setup(target_yaw=np.pi + np.radians(30))   # sees green, aims off to the side
    _, r, _, info = g.step(fire(0))
    assert info["duels"][0]["kind"] == "duel" and r[0] == 5 + 7


def test_no_duel_hint_when_target_cannot_see_shooter_or_hints_off():
    g = _duel_setup(target_yaw=0.0)     # pink faces away
    _, r, _, _ = g.step(fire(0))
    assert r[0] == 5
    g = _duel_setup(hints=False)
    _, r, _, _ = g.step(fire(0))
    assert r[0] == 5


def test_no_duel_hint_if_target_fired_first():
    g = _duel_setup(target_yaw=np.pi + np.radians(30))
    g.cooldown[4] = 20                  # pink fired (and missed) 3 s ago
    _, r, _, _ = g.step(fire(0))
    assert r[0] == 5
