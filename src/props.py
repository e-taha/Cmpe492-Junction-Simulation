"""
Roadside props for MuJoCo straight-road scenes.

Call add_roadside_props(spec, y_min, y_max) on an MjSpec *before* compile()
to scatter trees and road signs on both sides of the road.

Road layout assumed:
  Tarmac edge lines at x = ±3.5 m (7 m total width)
  Trees placed at |x| ∈ [4.8, 8.5] m (behind the verge)
  Signs placed at |x| ∈ [4.2, 5.5] m (shoulder)
"""

import math
import numpy as np
import mujoco


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def _quat_axis_angle(ax, ay, az, angle_rad):
    """Unit-quaternion [w, x, y, z] for rotation `angle_rad` around (ax,ay,az)."""
    h = angle_rad / 2.0
    s = math.sin(h)
    return [math.cos(h), ax * s, ay * s, az * s]


# ── Prop builders ──────────────────────────────────────────────────────────────

def _add_tree(wb, name: str, x: float, y: float,
              trunk_r: float, canopy_r: float, green: float) -> None:
    """Deciduous tree: brown cylinder trunk + green sphere canopy."""
    body = wb.add_body()
    body.name = name
    body.pos = [x, y, 0.0]

    trunk = body.add_geom()
    trunk.name = f"{name}_trunk"
    trunk.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    trunk.size = [trunk_r, 1.1, 0.0]           # half-height = 1.1 → z=0..2.2 m
    trunk.pos  = [0.0, 0.0, 1.1]
    trunk.rgba = [0.38, 0.24, 0.11, 1.0]
    trunk.contype = 0
    trunk.conaffinity = 0

    canopy = body.add_geom()
    canopy.name = f"{name}_canopy"
    canopy.type = mujoco.mjtGeom.mjGEOM_SPHERE
    canopy.size = [canopy_r, 0.0, 0.0]
    canopy.pos  = [0.0, 0.0, 2.2 + canopy_r]
    canopy.rgba = [0.14, green, 0.10, 1.0]
    canopy.contype = 0
    canopy.conaffinity = 0


def _add_speed_sign(wb, name: str, x: float, y: float) -> None:
    """Speed-limit sign: grey post + red-border disc + white face disc.
    Both disc geoms lie in the x-z plane (visible from north/south)."""
    body = wb.add_body()
    body.name = name
    body.pos = [x, y, 0.0]

    # Vertical post
    post = body.add_geom()
    post.name = f"{name}_post"
    post.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    post.size = [0.04, 1.3, 0.0]
    post.pos  = [0.0, 0.0, 1.3]
    post.rgba = [0.65, 0.65, 0.65, 1.0]
    post.contype = 0
    post.conaffinity = 0

    # Red border disc (cylinder with axis along y so the flat face is in x-z)
    q = _quat_axis_angle(1.0, 0.0, 0.0, math.pi / 2)   # 90° around x
    border = body.add_geom()
    border.name = f"{name}_border"
    border.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    border.size = [0.27, 0.022, 0.0]
    border.pos  = [0.0, 0.0, 2.75]
    border.quat = q
    border.rgba = [0.85, 0.10, 0.10, 1.0]
    border.contype = 0
    border.conaffinity = 0

    # White face disc (slightly in front of border)
    face = body.add_geom()
    face.name = f"{name}_face"
    face.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    face.size = [0.21, 0.025, 0.0]
    face.pos  = [0.0, 0.0, 2.75]
    face.quat = q
    face.rgba = [1.0, 1.0, 1.0, 1.0]
    face.contype = 0
    face.conaffinity = 0


def _add_warning_sign(wb, name: str, x: float, y: float) -> None:
    """Triangle warning sign: grey post + yellow diamond panel (box rotated 45°).
    Panel lies in the x-z plane, rotated 45° around y to form a diamond."""
    body = wb.add_body()
    body.name = name
    body.pos = [x, y, 0.0]

    # Vertical post
    post = body.add_geom()
    post.name = f"{name}_post"
    post.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    post.size = [0.04, 1.3, 0.0]
    post.pos  = [0.0, 0.0, 1.3]
    post.rgba = [0.65, 0.65, 0.65, 1.0]
    post.contype = 0
    post.conaffinity = 0

    # Yellow diamond — box in x-z plane, rotated 45° around y
    q_diamond = _quat_axis_angle(0.0, 1.0, 0.0, math.pi / 4)
    panel = body.add_geom()
    panel.name = f"{name}_panel"
    panel.type = mujoco.mjtGeom.mjGEOM_BOX
    panel.size = [0.24, 0.025, 0.24]
    panel.pos  = [0.0, 0.0, 2.75]
    panel.quat = q_diamond
    panel.rgba = [1.0, 0.82, 0.0, 1.0]
    panel.contype = 0
    panel.conaffinity = 0

    # Dark-red border box (slightly larger, same rotation, behind panel)
    border = body.add_geom()
    border.name = f"{name}_border"
    border.type = mujoco.mjtGeom.mjGEOM_BOX
    border.size = [0.28, 0.020, 0.28]
    border.pos  = [0.0, 0.0, 2.75]
    border.quat = q_diamond
    border.rgba = [0.75, 0.10, 0.05, 1.0]
    border.contype = 0
    border.conaffinity = 0


# ── Public API ─────────────────────────────────────────────────────────────────

def add_roadside_props(
    spec: mujoco.MjSpec,
    y_min: float,
    y_max: float,
    seed: int = 42,
    n_trees_per_side: int = 18,
    n_signs_per_side: int = 5,
) -> None:
    """
    Scatter trees and road signs on both sides of the road.

    Parameters
    ----------
    spec             : MjSpec to modify (before compile)
    y_min / y_max    : longitudinal extent of road to populate
    seed             : RNG seed for reproducible randomness
    n_trees_per_side : number of trees placed on each side
    n_signs_per_side : number of signs placed on each side
    """
    rng = np.random.default_rng(seed)
    wb  = spec.worldbody
    y_span = y_max - y_min

    for side_sign, side_tag in [(+1, "E"), (-1, "W")]:
        # ── Trees ──────────────────────────────────────────────────────────────
        # Evenly-spaced base positions with random jitter, then random x offset
        base_ys = np.linspace(y_min + 8.0, y_max - 8.0, n_trees_per_side)
        for i, base_y in enumerate(base_ys):
            y = float(base_y + rng.uniform(-y_span / (2 * n_trees_per_side), y_span / (2 * n_trees_per_side)))
            x = side_sign * float(rng.uniform(4.8, 8.5))
            trunk_r  = float(rng.uniform(0.10, 0.16))
            canopy_r = float(rng.uniform(1.1, 1.9))
            green    = float(rng.uniform(0.40, 0.65))
            _add_tree(wb, f"tree_{side_tag}_{i}", x, y, trunk_r, canopy_r, green)

        # ── Signs ──────────────────────────────────────────────────────────────
        # Evenly spaced, small y-jitter, alternate speed/warning
        sign_base_ys = np.linspace(y_min + 25.0, y_max - 25.0, n_signs_per_side)
        for i, base_y in enumerate(sign_base_ys):
            y = float(base_y + rng.uniform(-6.0, 6.0))
            x = side_sign * float(rng.uniform(4.2, 5.4))
            if i % 2 == 0:
                _add_speed_sign(wb, f"sign_{side_tag}_{i}", x, y)
            else:
                _add_warning_sign(wb, f"sign_{side_tag}_{i}", x, y)


def add_junction_props(
    spec: mujoco.MjSpec,
    seed: int = 42,
    road_half: float = 7.0,
    arm_len: float = 60.0,
) -> None:
    """
    Scatter trees and road signs around a 4-way junction scene.

    Trees fill the four corner quadrants (between road arms) and line the
    outer edges of each arm.  Signs are placed along the south arm (Ego
    approach) and north arm (Target approach).

    Parameters
    ----------
    spec      : MjSpec to modify (before compile)
    seed      : RNG seed
    road_half : half-width of each road arm in metres (default 7 m)
    arm_len   : length of each road arm beyond the junction box (default 60 m)
    """
    rng = np.random.default_rng(seed)
    wb  = spec.worldbody
    r   = road_half
    lo  = r + 1.5      # inner edge of tree/sign band
    hi  = r + 6.0      # outer edge of tree/sign band

    # ── Corner quadrant trees (fills the "open land" between arms) ─────────────
    for sx, sy, ctag in [(+1, +1, "NE"), (-1, +1, "NW"),
                          (-1, -1, "SW"), (+1, -1, "SE")]:
        for i in range(8):
            x = sx * float(rng.uniform(r + 1.5, r + 22.0))
            y = sy * float(rng.uniform(r + 1.5, r + 22.0))
            _add_tree(wb, f"tree_c{ctag}_{i}", x, y,
                      float(rng.uniform(0.10, 0.16)),
                      float(rng.uniform(1.0,  1.8)),
                      float(rng.uniform(0.40, 0.65)))

    # ── Trees along North / South arm outer edges ──────────────────────────────
    for sy, atag in [(+1, "N"), (-1, "S")]:
        for sx, stag in [(+1, "E"), (-1, "W")]:
            base_ys = np.linspace(r + 4.0, arm_len - 4.0, 9)
            for i, base_y in enumerate(base_ys):
                y = sy * float(base_y + rng.uniform(-2.5, 2.5))
                x = sx * float(rng.uniform(lo, hi))
                _add_tree(wb, f"tree_ns{atag}{stag}_{i}", x, y,
                          float(rng.uniform(0.10, 0.15)),
                          float(rng.uniform(1.0,  1.7)),
                          float(rng.uniform(0.40, 0.60)))

    # ── Trees along East / West arm outer edges ────────────────────────────────
    for sx, atag in [(+1, "E"), (-1, "W")]:
        for sy, stag in [(+1, "N"), (-1, "S")]:
            base_xs = np.linspace(r + 4.0, arm_len - 4.0, 7)
            for i, base_x in enumerate(base_xs):
                x = sx * float(base_x + rng.uniform(-2.5, 2.5))
                y = sy * float(rng.uniform(lo, hi))
                _add_tree(wb, f"tree_ew{atag}{stag}_{i}", x, y,
                          float(rng.uniform(0.10, 0.15)),
                          float(rng.uniform(1.0,  1.7)),
                          float(rng.uniform(0.40, 0.60)))

    # ── Signs along South arm (Ego approach) — east side ──────────────────────
    for i, base_y in enumerate(np.linspace(-arm_len + 10.0, -r - 4.0, 4)):
        y = float(base_y + rng.uniform(-3.0, 3.0))
        x = float(rng.uniform(r + 1.0, r + 3.5))
        if i % 2 == 0:
            _add_speed_sign(wb, f"sign_SE_{i}", x, y)
        else:
            _add_warning_sign(wb, f"sign_SE_{i}", x, y)

    # ── Signs along North arm (Target approach) — west side ───────────────────
    for i, base_y in enumerate(np.linspace(r + 4.0, arm_len - 10.0, 4)):
        y = float(base_y + rng.uniform(-3.0, 3.0))
        x = -float(rng.uniform(r + 1.0, r + 3.5))
        if i % 2 == 0:
            _add_speed_sign(wb, f"sign_NW_{i}", x, y)
        else:
            _add_warning_sign(wb, f"sign_NW_{i}", x, y)
