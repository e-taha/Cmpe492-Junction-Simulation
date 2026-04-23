"""
Pure Pursuit lateral controller and reference path utilities.

Path representation
-------------------
A path is a dense (N, 2) NumPy array of (x, y) waypoints in the world frame.
Use the builder functions below to construct straight segments and circular
arcs, then join them with make_path().

MuJoCo steering sign convention (simple_car.xml)
-------------------------------------------------
  tendon "steering_link"  =  -fl_angle  -  fr_angle
  The position actuator drives the tendon to the ctrl value.
  With fl_angle ≈ fr_angle ≈ delta:
      tendon  ≈  -2 * delta
      ctrl    =  -2 * delta       (delta > 0 → left turn, ctrl < 0)
                                  (delta < 0 → right turn, ctrl > 0)

Heading convention
------------------
  Yaw heading is derived from d.xmat[body_id]:
      d.xmat has shape (nbody, 9) — each row is a row-major flattened 3×3
      rotation matrix R.  The body's local +x axis (forward) in world frame
      is the first column of R:
          world_fx = R[0,0] = xmat[body_id, 0]
          world_fy = R[1,0] = xmat[body_id, 3]
      heading = atan2(world_fy, world_fx)
"""

import math
import numpy as np


# ── Path builders ──────────────────────────────────────────────────────────────

def straight_segment(x0: float, y0: float, x1: float, y1: float,
                     spacing: float = 0.25) -> np.ndarray:
    """Evenly spaced waypoints along a straight line from (x0,y0) to (x1,y1)."""
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(dist / spacing) + 1)
    return np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])


def arc_segment(cx: float, cy: float, r: float,
                theta_start: float, theta_end: float,
                spacing: float = 0.25) -> np.ndarray:
    """
    Evenly spaced waypoints along a circular arc.

    Parameters
    ----------
    cx, cy       : arc centre in world frame [m]
    r            : arc radius [m]
    theta_start  : start angle [rad], measured from positive x-axis
    theta_end    : end angle [rad]  — clockwise turn → theta_end < theta_start
    spacing      : approximate distance between waypoints [m]
    """
    arc_len = abs(theta_end - theta_start) * r
    n = max(2, int(arc_len / spacing) + 1)
    thetas = np.linspace(theta_start, theta_end, n)
    return np.column_stack([cx + r * np.cos(thetas),
                            cy + r * np.sin(thetas)])


def make_path(*segments: np.ndarray) -> np.ndarray:
    """
    Concatenate path segments into a single (N, 2) waypoint array.
    Removes the duplicate boundary point at each junction.
    """
    parts = [segments[0]]
    for seg in segments[1:]:
        parts.append(seg[1:])   # drop first point — same as last of previous
    return np.vstack(parts)


def heading_from_xmat(xmat: np.ndarray, body_id: int) -> float:
    """
    Extract yaw heading [rad] of a MuJoCo body from d.xmat.

    d.xmat shape: (nbody, 9) — row-major flattened 3×3 rotation matrices.
    Body forward (+x) in world frame: first column of R → [xmat[i,0], xmat[i,3]].
    """
    return math.atan2(float(xmat[body_id, 3]), float(xmat[body_id, 0]))


# ── Pure Pursuit controller ────────────────────────────────────────────────────

class PurePursuit:
    """
    Pure Pursuit lateral controller.

    At each timestep call compute_ctrl(x, y, heading) to get the MuJoCo
    steering actuator ctrl value.

    Parameters
    ----------
    path_xy      : (N, 2) reference waypoints [m]
    wheelbase    : vehicle wheelbase L [m]
    lookahead    : lookahead distance L_d [m]
    max_delta    : maximum wheel steering angle [rad], used to clamp output
    """

    CTRL_LIMIT = 1.2217   # rad — matches simple_car.xml ctrlrange

    def __init__(self, path_xy: np.ndarray, wheelbase: float,
                 lookahead: float, max_delta: float = 0.610865):
        self.path      = np.asarray(path_xy, dtype=float)
        self.L         = wheelbase
        self.ld        = lookahead
        self.max_delta = max_delta
        self._near     = 0          # monotonic search index

    # ── public ────────────────────────────────────────────────────────────────

    def compute_ctrl(self, x: float, y: float, heading: float) -> float:
        """
        Compute steering ctrl value for the current vehicle state.

        Returns a value in [-CTRL_LIMIT, +CTRL_LIMIT].
        Negative → left turn;  positive → right turn.

        Parameters
        ----------
        x, y    : chassis world position [m]
        heading : chassis yaw [rad] from heading_from_xmat()
        """
        self._near = self._nearest_idx(x, y)
        lx, ly    = self._lookahead_point(x, y)

        # Signed angle from heading to lookahead direction (vehicle frame)
        alpha = math.atan2(ly - y, lx - x) - heading
        alpha = (alpha + math.pi) % (2.0 * math.pi) - math.pi  # wrap [-π, π]

        # Pure Pursuit curvature κ = 2 sin(α) / L_d
        kappa = 2.0 * math.sin(alpha) / self.ld
        delta = math.atan(kappa * self.L)
        delta = max(-self.max_delta, min(self.max_delta, delta))

        # MuJoCo ctrl = -2 * delta
        ctrl = -2.0 * delta
        return max(-self.CTRL_LIMIT, min(self.CTRL_LIMIT, ctrl))

    def lookahead_point(self, x: float, y: float) -> tuple:
        """Return the current lookahead (x, y) — useful for visualization."""
        self._near = self._nearest_idx(x, y)
        return self._lookahead_point(x, y)

    # ── private ────────────────────────────────────────────────────────────────

    def _nearest_idx(self, x: float, y: float) -> int:
        """Find index of closest waypoint, searching a window around _near."""
        pos   = np.array([x, y])
        start = max(0, self._near - 5)
        end   = min(len(self.path), self._near + 60)
        dists = np.linalg.norm(self.path[start:end] - pos, axis=1)
        return start + int(np.argmin(dists))

    def _lookahead_point(self, x: float, y: float) -> tuple:
        """First waypoint at least ld away from (x, y), starting from _near."""
        for i in range(self._near, len(self.path)):
            if math.hypot(self.path[i, 0] - x, self.path[i, 1] - y) >= self.ld:
                return float(self.path[i, 0]), float(self.path[i, 1])
        # Past end of path — clamp to last waypoint
        return float(self.path[-1, 0]), float(self.path[-1, 1])


# ── Viewer overlay helpers ─────────────────────────────────────────────────────

def draw_path_geoms(viewer, path: np.ndarray, color,
                    lookahead_xy: tuple,
                    label_pos: np.ndarray,
                    label_text: str,
                    start_idx: int = 0) -> int:
    """
    Draw one path (line segments + lookahead sphere + HUD label) into the
    viewer's user_scn geom buffer starting at start_idx.

    Returns the next free geom index.  Does NOT set viewer.user_scn.ngeom —
    the caller is responsible for that, which allows multiple paths to share
    the same buffer by chaining calls.

    Parameters
    ----------
    viewer        : passive viewer handle
    path          : (N, 2) waypoints in world frame
    color         : RGBA sequence [r, g, b, a] for the path lines
    lookahead_xy  : (lx, ly) current lookahead point (drawn as a sphere)
    label_pos     : 3-D world position for the HUD label
    label_text    : string to display above the vehicle
    start_idx     : first geom slot to write into
    """
    import mujoco

    PATH_Z      = 0.05
    PATH_STRIDE = 4
    LD_COLOR    = np.array([1.0, 0.8, 0.0, 1.0], dtype=np.float32)
    LABEL_COLOR = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    path_rgba   = np.array(color, dtype=np.float32)

    max_geom = viewer.user_scn.maxgeom
    idx = start_idx

    # ── Path line segments ─────────────────────────────────────────────────────
    i = 0
    while i + PATH_STRIDE < len(path) and idx < max_geom - 2:
        p1 = np.array([path[i,             0], path[i,             1], PATH_Z])
        p2 = np.array([path[i + PATH_STRIDE, 0], path[i + PATH_STRIDE, 1], PATH_Z])
        mujoco.mjv_connector(
            viewer.user_scn.geoms[idx],
            mujoco.mjtGeom.mjGEOM_LINE,
            2.0, p1, p2,
        )
        viewer.user_scn.geoms[idx].rgba[:] = path_rgba
        idx += 1
        i   += PATH_STRIDE

    # ── Lookahead sphere ───────────────────────────────────────────────────────
    if lookahead_xy is not None and idx < max_geom - 1:
        lx, ly = lookahead_xy
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.35, 0.35, 0.35]),
            np.array([lx, ly, 0.4]),
            np.eye(3).flatten(),
            LD_COLOR,
        )
        idx += 1

    # ── HUD label ──────────────────────────────────────────────────────────────
    if label_text is not None and idx < max_geom:
        geom = viewer.user_scn.geoms[idx]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LABEL,
            np.zeros(3),
            np.asarray(label_pos),
            np.eye(3).flatten(),
            LABEL_COLOR,
        )
        geom.label = label_text
        idx += 1

    return idx


def draw_overlay(viewer, path: np.ndarray,
                 lookahead_xy: tuple,
                 label_pos: np.ndarray,
                 label_text: str) -> None:
    """
    Convenience wrapper: draw a single path (green) and set ngeom.
    For multi-path scenes use draw_path_geoms() directly.
    """
    idx = draw_path_geoms(
        viewer, path,
        color=[0.0, 0.9, 0.2, 0.9],   # green
        lookahead_xy=lookahead_xy,
        label_pos=label_pos,
        label_text=label_text,
        start_idx=0,
    )
    viewer.user_scn.ngeom = idx
