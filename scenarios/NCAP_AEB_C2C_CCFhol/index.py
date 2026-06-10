"""
NCAP AEB C2C CCFhol — Car-to-Car Front Head-On Lane-Change Overlap (2023)
=========================================================================
Source scenario: NCAP_scenarios/NCAP_AEB_C2C_CCFhol_2023.xosc

All three road users are full Edgar vehicles (simple_car.xml):

  Ego  (northbound, x=+1.75 m)
       Straight path from y=0 to y=350.  Speed-controlled to 70 km/h.

  GVT  (southbound, x=-1.75 → x=+1.75 m)
       Path: straight south → cosine lane-change over 60 m → straight south
       in Ego's lane.  Speed-controlled to 70 km/h.

  SOV  (southbound, x=-1.75 m)
       Straight path south.  Speed-controlled to 70 km/h.
       Stays 19.4 m freespace ahead of GVT throughout.

Physics collision between GVT/SOV and Ego is disabled via contype bitmasks
so that the head-on approach does not destabilise the simulation.  A bounding-
box overlap check provides the "virtual collision" detection.

NCAP parameters (from .xosc)
-----------------------------
  Ego / GVT speed  : 70 km/h  (= 19.444 m/s)
  Lane-change dist : 60 m   (longitudinal, NCAP parameter L)
  Lane-change TTC  : 1.5 s  (TTC when lane change ends, NCAP parameter)
  SOV freespace    : 19.4 m (gap GVT→SOV front bumper, NCAP parameter F)
  Lateral offset   : 3.5 m  (= two half-lane widths)

Derived timing  (closing speed = 2 × 19.444 = 38.888 m/s)
----------------------------------------------------------
  LC duration   : 60 / 19.444 ≈ 3.086 s
  LC start TTC  : 3.086 + 1.5 = 4.586 s  before collision
  Gap at LC start: 4.586 × 38.888 ≈ 178.3 m  (between vehicle centres)
  With GVT_START_Y=250, EGO_START_Y=0:
    expected gap at t=0: 250 m  →  LC starts when gap has closed to 178.3 m

Collision safety
----------------
  After compiling the model, all geoms belonging to gvt-* or sov-* bodies are
  set to contype=2 / conaffinity=2, while Ego geoms remain at contype=1 /
  conaffinity=1.  The floor is set to contype=3 / conaffinity=3.  This means:
    floor ↔ Ego  : (3 & 1) > 0  → collide  ✓
    floor ↔ GVT  : (3 & 2) > 0  → collide  ✓
    Ego  ↔ GVT   : (1 & 2) = 0  → no physics collision  ✓
  A bounding-box overlap test reports the virtual collision time/speed.

Usage
-----
  mjpython scenarios/NCAP_AEB_C2C_CCFhol/index.py
  mjpython scenarios/NCAP_AEB_C2C_CCFhol/index.py --viewer
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ── Module path ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from pure_pursuit import (PurePursuit, straight_segment, arc_segment,
                          make_path, heading_from_xmat, draw_path_geoms)
from props import add_roadside_props

# ── File paths ─────────────────────────────────────────────────────────────────
SCENE_PATH  = ROOT_DIR / "scenes" / "ncap_straight_road.xml"
MODEL_PATH  = ROOT_DIR / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "ncap_ccfhol.csv"

# ── NCAP scenario parameters ───────────────────────────────────────────────────
EGO_SPEED_KPH       = 70.0
EGO_SPEED_MS        = EGO_SPEED_KPH / 3.6          # ≈ 19.444 m/s
GVT_SPEED_MS        = EGO_SPEED_MS

EGO_START_Y         = 0.0
GVT_START_Y         = 250.0
SOV_FREESPACE_M     = 19.4                          # bumper-to-bumper gap
GVT_HALF_LEN        = 4.973 / 2                    # same model as Ego
SOV_HALF_LEN        = 4.973 / 2
SOV_START_Y         = GVT_START_Y - (SOV_FREESPACE_M + GVT_HALF_LEN + SOV_HALF_LEN)
                      # ≈ 225.6 m  (centre-to-centre distance)

LANE_EGO_X          = 1.75
LANE_GVT_X          = -1.75
LANE_CHANGE_DIST_M  = 60.0
LANE_CHANGE_TTC_S   = 1.5

# ── Derived timing ─────────────────────────────────────────────────────────────
CLOSING_SPEED_MS    = EGO_SPEED_MS + GVT_SPEED_MS  # ≈ 38.888 m/s
LC_DURATION_S       = LANE_CHANGE_DIST_M / GVT_SPEED_MS   # ≈ 3.086 s
LC_START_TTC_S      = LC_DURATION_S + LANE_CHANGE_TTC_S   # ≈ 4.586 s
INITIAL_SEP_M       = GVT_START_Y - EGO_START_Y           # 250 m
TIME_TO_COLLISION_S = INITIAL_SEP_M / CLOSING_SPEED_MS    # ≈ 6.43 s

# GVT y-coordinate where lane change begins / ends (spatial, not temporal)
_LC_START_TTC_GAP   = LC_START_TTC_S * CLOSING_SPEED_MS   # ≈ 178.3 m
LC_START_Y          = GVT_START_Y - (INITIAL_SEP_M - _LC_START_TTC_GAP) / 2
# More precisely: at lane-change start, ego_y + gvt_y = 2×average, and
# gvt_y - ego_y = LC_START_TTC_GAP.  With equal acceleration from rest the
# vehicles meet at y ≈ 125 m.  For the spatial path we just encode the
# y-coordinates directly.
LC_START_Y          = GVT_START_Y - (INITIAL_SEP_M - _LC_START_TTC_GAP) * GVT_SPEED_MS / CLOSING_SPEED_MS
                      # = 250 - 35.85 ≈ 214.15 m
LC_END_Y            = LC_START_Y - LANE_CHANGE_DIST_M      # ≈ 154.15 m

# ── Vehicle constants ──────────────────────────────────────────────────────────
WHEELBASE_M         = 3.128
WHEEL_RADIUS_M      = 0.346
LOOKAHEAD_EGO_M     = 8.0
LOOKAHEAD_GVT_M     = 5.0   # shorter for tighter lane-change tracking
LOOKAHEAD_SOV_M     = 8.0
TEST_DURATION_S     = 15.0

# ── Collision detection geometry ──────────────────────────────────────────────
EGO_HALF_LEN        = 4.973 / 2    # ≈ 2.487 m  (chassis box size from XML)
EGO_HALF_WID        = 1.941 / 2    # ≈ 0.971 m
GVT_COL_HALF_LEN    = EGO_HALF_LEN
GVT_COL_HALF_WID    = EGO_HALF_WID

# ── Path colours for viewer ────────────────────────────────────────────────────
COLOR_EGO = [0.3, 0.5, 1.0, 0.9]   # blue
COLOR_GVT = [1.0, 0.3, 0.2, 0.9]   # red
COLOR_SOV = [1.0, 0.6, 0.1, 0.9]   # orange


# ── Reference paths ────────────────────────────────────────────────────────────

def _build_gvt_path() -> np.ndarray:
    """
    GVT path: straight south → cosine lane-change (60 m) → straight south
    in Ego's lane.  The cosine S-curve gives smooth lateral acceleration.
    """
    pre_lc = straight_segment(LANE_GVT_X, GVT_START_Y,
                               LANE_GVT_X, LC_START_Y)

    n_pts = 40
    lc_pts = []
    for i in range(n_pts + 1):
        p     = i / n_pts
        alpha = (1.0 - math.cos(math.pi * p)) / 2.0
        lc_pts.append([
            LANE_GVT_X + alpha * (LANE_EGO_X - LANE_GVT_X),
            LC_START_Y - p * LANE_CHANGE_DIST_M,
        ])
    lc_array = np.array(lc_pts)

    post_lc = straight_segment(LANE_EGO_X, LC_END_Y, LANE_EGO_X, -100.0)

    return make_path(pre_lc, lc_array, post_lc)


PATH_EGO = straight_segment(LANE_EGO_X, -5.0, LANE_EGO_X, 350.0)
PATH_GVT = _build_gvt_path()
PATH_SOV = straight_segment(LANE_GVT_X, SOV_START_Y, LANE_GVT_X, -100.0)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> mujoco.MjModel:
    scene_spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    spec_ego   = mujoco.MjSpec.from_file(str(MODEL_PATH))
    spec_gvt   = mujoco.MjSpec.from_file(str(MODEL_PATH))
    spec_sov   = mujoco.MjSpec.from_file(str(MODEL_PATH))

    add_roadside_props(scene_spec, y_min=-10.0, y_max=260.0, seed=7)

    scene_spec.attach(spec_ego, frame="spawn_ego", prefix="ego-")
    scene_spec.attach(spec_gvt, frame="spawn_gvt", prefix="gvt-")
    scene_spec.attach(spec_sov, frame="spawn_sov", prefix="sov-")

    m = scene_spec.compile()

    # Assign contype bitmasks so that GVT/SOV do not physically collide with
    # Ego but still interact with the floor (see module docstring).
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        m.geom_contype[floor_id]     = 3
        m.geom_conaffinity[floor_id] = 3

    for i in range(m.ngeom):
        body_id   = int(m.geom_bodyid[i])
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith(("gvt-", "sov-")):
            m.geom_contype[i]     = 2
            m.geom_conaffinity[i] = 2

    return m


# ── Vehicle helpers ────────────────────────────────────────────────────────────

def _vehicle_ids(m: mujoco.MjModel, prefix: str) -> dict:
    def _aid(name): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    def _sid(name): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   name)
    def _bid(name): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     name)
    def _jid(name): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,    name)
    return {
        "steering":  _aid(f"{prefix}steering"),
        "throttle":  _aid(f"{prefix}throttle"),
        "brake":     _aid(f"{prefix}brake"),
        "vel":       _sid(f"{prefix}velocimeter"),
        "acc":       _sid(f"{prefix}accelerometer"),
        "chassis":   _bid(f"{prefix}chassis"),
        "root":      _jid(f"{prefix}root"),
        "fl_spin":   _jid(f"{prefix}fl_spin"),
        "fr_spin":   _jid(f"{prefix}fr_spin"),
        "rl_spin":   _jid(f"{prefix}rl_spin"),
        "rr_spin":   _jid(f"{prefix}rr_spin"),
        "fl_steer":  _jid(f"{prefix}wheel_fl_steering"),
    }


def _init_vehicle_speed(m: mujoco.MjModel, d: mujoco.MjData,
                        ids: dict, speed_ms: float, direction: float) -> None:
    """
    Pre-set freejoint and wheel-spin velocities so vehicles start already
    at target speed, avoiding a large initial slip transient.

    direction: +1 northbound (Ego),  -1 southbound (GVT / SOV)
    """
    # Freejoint: 6 DOF — linear [vx, vy, vz], angular [ωx, ωy, ωz]
    root_dof = int(m.jnt_dofadr[ids["root"]])
    d.qvel[root_dof + 1] = speed_ms * direction   # world-frame vy

    # Wheel spin: ω = v / r  (positive = forward for any heading)
    omega = speed_ms / WHEEL_RADIUS_M
    for jname in ("fl_spin", "fr_spin", "rl_spin", "rr_spin"):
        dof = int(m.jnt_dofadr[ids[jname]])
        d.qvel[dof] = omega


def _step_vehicle(m: mujoco.MjModel, d: mujoco.MjData, ids: dict,
                  tracker: PurePursuit, target_speed: float) -> None:
    """Apply steering + speed-control actuator outputs for one vehicle."""
    chassis = ids["chassis"]
    cx  = float(d.xpos[chassis][0])
    cy  = float(d.xpos[chassis][1])
    hdg = heading_from_xmat(d.xmat, chassis)

    vel_adr = int(m.sensor_adr[ids["vel"]])
    vx  = float(d.sensordata[vel_adr])
    vy  = float(d.sensordata[vel_adr + 1])
    spd = math.sqrt(vx**2 + vy**2)

    speed_err = target_speed - spd
    throttle  = max(0.0, min(1.0, 0.15 + 0.6 * speed_err))

    d.ctrl[ids["steering"]] = tracker.compute_ctrl(cx, cy, hdg)
    d.ctrl[ids["throttle"]] = throttle
    d.ctrl[ids["brake"]]    = 0.0


# ── Collision detection ────────────────────────────────────────────────────────

def _check_collision(ego_x: float, ego_y: float,
                     gvt_x: float, gvt_y: float) -> bool:
    lon = abs(ego_y - gvt_y) < (EGO_HALF_LEN + GVT_COL_HALF_LEN)
    lat = abs(ego_x - gvt_x) < (EGO_HALF_WID + GVT_COL_HALF_WID)
    return lon and lat


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel, show_viewer: bool = False) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    ids_ego = _vehicle_ids(m, "ego-")
    ids_gvt = _vehicle_ids(m, "gvt-")
    ids_sov = _vehicle_ids(m, "sov-")

    # Start all vehicles already at target speed (no ramp-up transient)
    _init_vehicle_speed(m, d, ids_ego, EGO_SPEED_MS, +1.0)
    _init_vehicle_speed(m, d, ids_gvt, GVT_SPEED_MS, -1.0)
    _init_vehicle_speed(m, d, ids_sov, GVT_SPEED_MS, -1.0)

    tracker_ego = PurePursuit(PATH_EGO, WHEELBASE_M, LOOKAHEAD_EGO_M)
    tracker_gvt = PurePursuit(PATH_GVT, WHEELBASE_M, LOOKAHEAD_GVT_M)
    tracker_sov = PurePursuit(PATH_SOV, WHEELBASE_M, LOOKAHEAD_SOV_M)

    times, ego_xs, ego_ys           = [], [], []
    ego_speeds, accel_lons, accel_lats = [], [], []
    steer_angles                    = []
    gvt_xs, gvt_ys                  = [], []
    separations, collision_flags    = [], []

    collision_time  = None
    collision_speed = None

    ego_vel_adr = int(m.sensor_adr[ids_ego["vel"]])
    ego_acc_adr = int(m.sensor_adr[ids_ego["acc"]])
    ego_fl_qpos = int(m.jnt_qposadr[ids_ego["fl_steer"]])

    def _record(gvt_x: float, gvt_y: float, collided: bool) -> None:
        pos = d.xpos[ids_ego["chassis"]]
        vx  = float(d.sensordata[ego_vel_adr])
        vy  = float(d.sensordata[ego_vel_adr + 1])
        spd = math.sqrt(vx**2 + vy**2)
        times.append(d.time)
        ego_xs.append(float(pos[0]))
        ego_ys.append(float(pos[1]))
        ego_speeds.append(spd)
        accel_lons.append(float(d.sensordata[ego_acc_adr]))
        accel_lats.append(float(d.sensordata[ego_acc_adr + 1]))
        steer_angles.append(float(d.qpos[ego_fl_qpos]))
        gvt_xs.append(gvt_x)
        gvt_ys.append(gvt_y)
        separations.append(abs(float(pos[1]) - gvt_y))
        collision_flags.append(int(collided))

    def _sim_step() -> tuple:
        """Advance all three vehicles by one timestep. Returns (gvt_x, gvt_y)."""
        _step_vehicle(m, d, ids_ego, tracker_ego, EGO_SPEED_MS)
        _step_vehicle(m, d, ids_gvt, tracker_gvt, GVT_SPEED_MS)
        _step_vehicle(m, d, ids_sov, tracker_sov, GVT_SPEED_MS)
        mujoco.mj_step(m, d)
        gvt_pos = d.xpos[ids_gvt["chassis"]]
        return float(gvt_pos[0]), float(gvt_pos[1])

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                t0 = time.time()
                gvt_x, gvt_y = _sim_step()

                ego_pos = d.xpos[ids_ego["chassis"]]
                ex, ey  = float(ego_pos[0]), float(ego_pos[1])
                collided = _check_collision(ex, ey, gvt_x, gvt_y)
                _record(gvt_x, gvt_y, collided)

                if collided and collision_time is None:
                    collision_time  = d.time
                    collision_speed = ego_speeds[-1]

                # Draw all three paths using the persistent trackers
                geom_idx = 0
                for (ids_v, tracker, path, color, name) in [
                    (ids_ego, tracker_ego, PATH_EGO, COLOR_EGO, "Ego"),
                    (ids_gvt, tracker_gvt, PATH_GVT, COLOR_GVT, "GVT"),
                    (ids_sov, tracker_sov, PATH_SOV, COLOR_SOV, "SOV"),
                ]:
                    cx  = float(d.xpos[ids_v["chassis"]][0])
                    cy  = float(d.xpos[ids_v["chassis"]][1])
                    lp  = tracker.lookahead_point(cx, cy)
                    lbl = d.xpos[ids_v["chassis"]].copy(); lbl[2] += 3.5
                    vel_a   = int(m.sensor_adr[ids_v["vel"]])
                    vx      = float(d.sensordata[vel_a])
                    vy2     = float(d.sensordata[vel_a + 1])
                    spd_kph = math.sqrt(vx**2 + vy2**2) * 3.6
                    geom_idx = draw_path_geoms(
                        viewer, path, color=color,
                        lookahead_xy=lp, label_pos=lbl,
                        label_text=f"{name} {spd_kph:.0f}km/h",
                        start_idx=geom_idx,
                    )

                # Add collision/TTC overlay on Ego label position
                gap  = gvt_y - ey
                ttc  = gap / CLOSING_SPEED_MS if gap > 0 else 0.0
                gvt_in_lane = gvt_x > 0.0
                if geom_idx < viewer.user_scn.maxgeom:
                    import mujoco as _mj
                    lbl_pos = d.xpos[ids_ego["chassis"]].copy(); lbl_pos[2] += 6.0
                    g = viewer.user_scn.geoms[geom_idx]
                    _mj.mjv_initGeom(g, _mj.mjtGeom.mjGEOM_LABEL,
                                     np.zeros(3), np.asarray(lbl_pos),
                                     np.eye(3).flatten(),
                                     np.array([1.,1.,0.,1.], dtype=np.float32))
                    g.label = (f"gap={gap:.1f}m  TTC={ttc:.2f}s  "
                               f"GVT={'EGO LANE' if gvt_in_lane else 'adj lane'}  "
                               f"{'*** COLLISION ***' if collided else ''}")
                    geom_idx += 1

                viewer.user_scn.ngeom = geom_idx
                viewer.sync()
                sleep_t = m.opt.timestep - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            gvt_x, gvt_y = _sim_step()
            ego_pos  = d.xpos[ids_ego["chassis"]]
            ex, ey   = float(ego_pos[0]), float(ego_pos[1])
            collided = _check_collision(ex, ey, gvt_x, gvt_y)
            _record(gvt_x, gvt_y, collided)
            if collided and collision_time is None:
                collision_time  = d.time
                collision_speed = ego_speeds[-1]

    return {
        "time_s":             np.array(times),
        "ego_x_m":            np.array(ego_xs),
        "ego_y_m":            np.array(ego_ys),
        "speed_ms":           np.array(ego_speeds),
        "accel_lon_ms2":      np.array(accel_lons),
        "accel_lat_ms2":      np.array(accel_lats),
        "steer_angle_rad":    np.array(steer_angles),
        "gvt_x_m":            np.array(gvt_xs),
        "gvt_y_m":            np.array(gvt_ys),
        "separation_m":       np.array(separations),
        "collision":          np.array(collision_flags),
        "collision_time_s":   collision_time,
        "collision_speed_ms": collision_speed,
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict:
    speeds  = data["speed_ms"]
    gvt_xs  = data["gvt_x_m"]
    gvt_ys  = data["gvt_y_m"]
    ego_ys  = data["ego_y_m"]

    gvt_in_ego_lane = bool(np.any(gvt_xs > 0.0))
    gvt_final_x     = float(gvt_xs[-1])

    lc_done = gvt_xs > 1.0
    if lc_done.any():
        idx         = int(np.argmax(lc_done))
        gap_at_end  = float(gvt_ys[idx]) - float(ego_ys[idx])
        ttc_at_lc_end = gap_at_end / CLOSING_SPEED_MS if gap_at_end > 0 else 0.0
    else:
        ttc_at_lc_end = float("nan")

    collision_detected = data["collision_time_s"] is not None
    collision_time     = data["collision_time_s"]   or float("nan")
    collision_speed_kph = (data["collision_speed_ms"] * 3.6
                           if data["collision_speed_ms"] is not None else float("nan"))

    return {
        "gvt_in_ego_lane":       gvt_in_ego_lane,
        "gvt_final_x_m":         gvt_final_x,
        "ttc_at_lc_end_s":       ttc_at_lc_end,
        "collision_detected":    collision_detected,
        "collision_time_s":      collision_time,
        "collision_speed_kph":   collision_speed_kph,
        "mean_ego_speed_kph":    float(np.mean(speeds)) * 3.6,
        "peak_ego_speed_kph":    float(np.max(speeds))  * 3.6,
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    ok = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")
    print(f"  [INFO] Mean Ego speed:           {metrics['mean_ego_speed_kph']:.1f} km/h")
    print(f"  [INFO] Peak Ego speed:           {metrics['peak_ego_speed_kph']:.1f} km/h")
    print(f"  [INFO] GVT final x:              {metrics['gvt_final_x_m']:.3f} m")
    if not math.isnan(metrics["ttc_at_lc_end_s"]):
        print(f"  [INFO] TTC when GVT enters lane: {metrics['ttc_at_lc_end_s']:.2f} s "
              f"(target ≈ {LANE_CHANGE_TTC_S:.1f} s)")
    if not math.isnan(metrics["collision_time_s"]):
        print(f"  [INFO] Collision time:           {metrics['collision_time_s']:.3f} s")
        print(f"  [INFO] Ego speed at collision:   {metrics['collision_speed_kph']:.1f} km/h")

    ok1 = 70.0 * 0.85 <= metrics["mean_ego_speed_kph"] <= 70.0 * 1.15
    ok  = ok and ok1
    print(f"\n  [{'PASS' if ok1 else 'FAIL'}] Ego maintains ~70 km/h "
          f"(mean = {metrics['mean_ego_speed_kph']:.1f} km/h)")

    ok2 = metrics["gvt_in_ego_lane"]
    ok  = ok and ok2
    print(f"\n  [{'PASS' if ok2 else 'FAIL'}] GVT completes lane change into Ego lane "
          f"(final GVT x = {metrics['gvt_final_x_m']:.3f} m, expected {LANE_EGO_X:.2f} m)")

    if math.isnan(metrics["ttc_at_lc_end_s"]):
        print("\n  [SKIP] TTC at lane change end: GVT never entered Ego lane")
    else:
        ok3 = abs(metrics["ttc_at_lc_end_s"] - LANE_CHANGE_TTC_S) <= 0.6
        ok  = ok and ok3
        print(f"\n  [{'PASS' if ok3 else 'FAIL'}] TTC at lane change end ≈ {LANE_CHANGE_TTC_S} s "
              f"(measured = {metrics['ttc_at_lc_end_s']:.2f} s)")

    ok4 = metrics["collision_detected"]
    ok  = ok and ok4
    print(f"\n  [{'PASS' if ok4 else 'FAIL'}] Collision detected "
          f"(expected without AEB — t={metrics['collision_time_s']:.2f} s, "
          f"v={metrics['collision_speed_kph']:.1f} km/h)")

    return ok


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([
        data["time_s"],
        data["ego_x_m"],     data["ego_y_m"],
        data["speed_ms"],    data["speed_ms"] * 3.6,
        data["accel_lon_ms2"], data["accel_lat_ms2"],
        data["steer_angle_rad"],
        data["gvt_x_m"],     data["gvt_y_m"],
        data["separation_m"], data["collision"],
    ])
    np.savetxt(CSV_PATH, rows, delimiter=",",
               header="time_s,ego_x_m,ego_y_m,ego_speed_ms,ego_speed_kph,"
                      "accel_lon_ms2,accel_lat_ms2,steer_angle_rad,"
                      "gvt_x_m,gvt_y_m,longitudinal_sep_m,collision",
               comments="")
    print(f"\n  CSV saved → {CSV_PATH}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NCAP AEB C2C CCFhol — head-on cut-in, 3 Edgar vehicles")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open MuJoCo viewer during simulation")
    args = parser.parse_args()

    print("=" * 70)
    print("  NCAP AEB C2C CCFhol — Car-to-Car Front Head-On Cut-In Overlap")
    print("=" * 70)
    print(f"  Scene:               {SCENE_PATH}")
    print(f"  Model:               {MODEL_PATH}")
    print(f"  Ego speed:           {EGO_SPEED_KPH} km/h  (northbound, x={LANE_EGO_X} m)")
    print(f"  GVT speed:           {EGO_SPEED_KPH} km/h  (southbound, x={LANE_GVT_X}→{LANE_EGO_X} m)")
    print(f"  SOV speed:           {EGO_SPEED_KPH} km/h  (southbound, x={LANE_GVT_X} m)")
    print(f"  GVT start:           y={GVT_START_Y} m")
    print(f"  SOV start:           y={SOV_START_Y:.1f} m")
    print(f"  LC spatial start:    GVT y={LC_START_Y:.2f} m")
    print(f"  LC spatial end:      GVT y={LC_END_Y:.2f} m")
    print(f"  Expected TTC at LC end:  {LANE_CHANGE_TTC_S} s")
    print(f"  Expected collision:  t≈{TIME_TO_COLLISION_S:.2f} s")
    print(f"  Viewer:              {'on' if args.viewer else 'off (pass --viewer)'}")

    for p, label in [(SCENE_PATH, "scene"), (MODEL_PATH, "model")]:
        if not p.exists():
            print(f"\n  ERROR: {label} not found: {p}")
            sys.exit(1)

    m = build_model()
    print("\n  Running simulation...")
    data    = run_simulation(m, show_viewer=args.viewer)
    metrics = extract_metrics(data)
    ok      = print_report(metrics)
    save_csv(data)

    print("\n" + "=" * 70)
    print(f"  {'PASS' if ok else 'FAIL'} — NCAP AEB C2C CCFhol")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
