"""Arena geometry: everything is an axis-aligned box (walls, cover, platforms, stairs).

Coordinates are metres. x runs 0..30 (green spawns at the low-x end, pink at the high-x end),
y runs 0..20, z is up. The layout is point-symmetric around the centre (15, 10)
so neither team has a map advantage.
"""
from __future__ import annotations

import numpy as np
from numba import njit

WIDTH, DEPTH, WALL_H = 30.0, 20.0, 3.0
STEP_H = 0.3  # tallest ledge an agent can walk up

# Box kinds (used by the replay viewer later)
FLOOR, WALL, COVER, PLATFORM, STAIR = range(5)


def _stairs(x0, x1, y0, y1, axis, direction, top=1.5, n=5, run=0.4):
    """Stairs leading down from a platform edge. axis 0 = along x, 1 = along y."""
    boxes = []
    for k in range(1, n + 1):
        h = top - top / (n + 1) * k
        a = (x0, y0)[axis] + direction * run * (k - 1)
        b = a + direction * run
        lo, hi = min(a, b), max(a, b)
        if axis == 0:
            boxes.append(((lo, y0, 0), (hi, y1, h), STAIR))
        else:
            boxes.append(((x0, lo, 0), (x1, hi, h), STAIR))
    return boxes


def _green_half():
    """Boxes for the green half; the pink half is the point mirror."""
    b = []
    # Raised platform on the green half, stairs heading toward the centre.
    b.append(((5, 14, 0), (8, 18, 1.5), PLATFORM))
    b += _stairs(8, None, 14, 16, axis=0, direction=+1)
    # Full-height walls that break long sightlines.
    b.append(((10, 4, 0), (10.3, 9, WALL_H), WALL))
    b.append(((4, 6, 0), (7, 6.3, WALL_H), WALL))
    # Waist-high cover (1 m): crouch behind it, stand to shoot over it.
    b.append(((3, 10, 0), (4, 11.5, 1.0), COVER))
    b.append(((8, 10, 0), (9.5, 11, 1.0), COVER))
    b.append(((11.5, 15, 0), (12.5, 16.5, 1.0), COVER))
    b.append(((6, 2, 0), (7.5, 3, 1.0), COVER))
    # Stairs on one side of the central tower (the mirror adds the other side).
    b += _stairs(14.25, 15.75, 8.5, None, axis=1, direction=-1)
    return b


def _mirror(box):
    (x0, y0, z0), (x1, y1, z1), kind = box
    return (WIDTH - x1, DEPTH - y1, z0), (WIDTH - x0, DEPTH - y0, z1), kind


def build_layout():
    t = 0.3
    boxes = [
        ((-50, -50, -0.1), (80, 70, 0.0), FLOOR),
        ((-t, -t, 0), (0, DEPTH + t, WALL_H), WALL),
        ((WIDTH, -t, 0), (WIDTH + t, DEPTH + t, WALL_H), WALL),
        ((0, -t, 0), (WIDTH, 0, WALL_H), WALL),
        ((0, DEPTH, 0), (WIDTH, DEPTH + t, WALL_H), WALL),
        ((13.5, 8.5, 0), (16.5, 11.5, 1.5), PLATFORM),  # central tower
    ]
    half = _green_half()
    boxes += half + [_mirror(x) for x in half]
    return boxes


# No bases: each team (re)spawns at a random spot along its own end of the arena,
# so there is nothing fixed to defend. Green spawns along the low-x edge facing +x,
# pink along the high-x edge facing -x.
SPAWN_STRIP = (1.0, 3.0)   # distance from the end wall, metres


def ray_boxes(origins, dirs, lo, hi):
    """Nearest hit of each ray against boxes (slab method, vectorised).

    origins, dirs: (R, 3). lo, hi: (B, 3). Returns (t, index) per ray;
    t is inf and index -1 where nothing is hit. Rays starting inside a box hit at t=0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs[:, None, :]
        t1 = (lo[None] - origins[:, None, :]) * inv
        t2 = (hi[None] - origins[:, None, :]) * inv
    t1 = np.nan_to_num(t1, nan=-np.inf)
    t2 = np.nan_to_num(t2, nan=np.inf)
    tnear = np.minimum(t1, t2).max(axis=2)
    tfar = np.maximum(t1, t2).min(axis=2)
    hit = (tfar >= np.maximum(tnear, 0.0))
    t = np.where(hit, np.maximum(tnear, 0.0), np.inf)
    idx = t.argmin(axis=1)
    best = t[np.arange(len(t)), idx]
    idx = np.where(np.isfinite(best), idx, -1)
    return best, idx


@njit(cache=True, fastmath=True)
def _slab(ox, oy, oz, ix, iy, iz, lx, ly, lz, hx, hy, hz, best):
    t1 = (lx - ox) * ix; t2 = (hx - ox) * ix
    tn = min(t1, t2); tf = max(t1, t2)
    t1 = (ly - oy) * iy; t2 = (hy - oy) * iy
    tn = max(tn, min(t1, t2)); tf = min(tf, max(t1, t2))
    t1 = (lz - oz) * iz; t2 = (hz - oz) * iz
    tn = max(tn, min(t1, t2)); tf = min(tf, max(t1, t2))
    if tn < 0.0:
        tn = 0.0
    if tf >= tn and tn < best:
        return tn
    return best


@njit(cache=True, fastmath=True)
def trace_rays(origins, dirs, owners, lo, hi, alo, ahi):
    """Nearest hit per ray against world boxes (lo, hi) and agent boxes (alo, ahi).

    The ray's owner (agent index, or -1) is skipped. Returns (t, agent) where agent
    is -1 if the world was hit first (or nothing: t = inf)."""
    R = origins.shape[0]
    out_t = np.empty(R)
    out_a = np.empty(R, dtype=np.int64)
    for r in range(R):
        ox, oy, oz = origins[r, 0], origins[r, 1], origins[r, 2]
        dx, dy, dz = dirs[r, 0], dirs[r, 1], dirs[r, 2]
        ix = 1.0 / dx if dx != 0.0 else 1e30
        iy = 1.0 / dy if dy != 0.0 else 1e30
        iz = 1.0 / dz if dz != 0.0 else 1e30
        best = np.inf
        for b in range(lo.shape[0]):
            best = _slab(ox, oy, oz, ix, iy, iz, lo[b, 0], lo[b, 1], lo[b, 2], hi[b, 0], hi[b, 1], hi[b, 2], best)
        agent = -1
        for a in range(alo.shape[0]):
            if a == owners[r]:
                continue
            t = _slab(ox, oy, oz, ix, iy, iz, alo[a, 0], alo[a, 1], alo[a, 2], ahi[a, 0], ahi[a, 1], ahi[a, 2], best)
            if t < best:
                best = t
                agent = a
        out_t[r] = best
        out_a[r] = agent
    return out_t, out_a


_NO_AGENTS = np.zeros((0, 3))


class Arena:
    def __init__(self):
        self.boxes = build_layout()
        self.lo = np.array([b[0] for b in self.boxes], dtype=np.float64)
        self.hi = np.array([b[1] for b in self.boxes], dtype=np.float64)
        self.kinds = np.array([b[2] for b in self.boxes])

    def random_spawn(self, team, rng, radius=0.3):
        """A free floor spot on the team's end of the arena (team 0 green, 1 pink)."""
        while True:
            x = rng.uniform(*SPAWN_STRIP)
            y = rng.uniform(1.0, DEPTH - 1.0)
            if team == 1:
                x = WIDTH - x
            lo, hi = self.lo[1:], self.hi[1:]  # skip floor
            if not ((lo[:, 0] < x + radius) & (hi[:, 0] > x - radius) &
                    (lo[:, 1] < y + radius) & (hi[:, 1] > y - radius)).any():
                return x, y, (0.0 if team == 0 else np.pi)

    def raycast(self, origins, dirs):
        o, d = np.atleast_2d(origins).astype(np.float64), np.atleast_2d(dirs).astype(np.float64)
        t, _ = trace_rays(o, d, np.full(len(o), -1), self.lo, self.hi, _NO_AGENTS, _NO_AGENTS)
        return t, None

    def try_move(self, x, y, feet, nx, ny, radius, height):
        """Move a cylinder-ish body from (x, y) to (nx, ny). Returns (x, y, feet) after
        collisions: blocked by anything taller than a step, otherwise it lands on the
        highest walkable surface under it (walking off a platform drops you down)."""
        lo, hi = self.lo, self.hi
        over = ((lo[:, 0] < nx + radius) & (hi[:, 0] > nx - radius) &
                (lo[:, 1] < ny + radius) & (hi[:, 1] > ny - radius))
        tops = hi[over, 2]
        bottoms = lo[over, 2]
        blocking = (tops > feet + STEP_H) & (bottoms < feet + height)
        if blocking.any():
            return x, y, feet
        walkable = tops[tops <= feet + STEP_H]
        return nx, ny, float(walkable.max()) if len(walkable) else 0.0
