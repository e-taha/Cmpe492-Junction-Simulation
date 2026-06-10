"""
NCAP AEB C2C CCFtap — Car-to-Car Front Turning Across Path (2023)
=================================================================
Source scenario: NCAP_scenarios/NCAP_AEB_C2C_CCFtap_2023.xosc

Ego (northbound) turns LEFT (farside/across-path) at a 4-way junction.
Target (southbound) travels straight through.  Without AEB they collide
inside the junction box when Ego's arc crosses Target's lane (x=−1.75 m).

Road layout (ncap_junction.xml)
--------------------------------
  N-S road : northbound x=+1.75, southbound x=−1.75
  E-W road : eastbound  y=−1.75, westbound  y=+1.75
  Junction box: ±7 m in x and y

Ego path
--------
  Segment 1 — straight approach, south arm:
    (1.75, -20) → (1.75, -7)           heading north
  Segment 2 — left turn (CCW arc):
    centre (−7, −7), R=8.75 m, θ=0→π/2
    entry  (1.75, -7)   heading north
    exit   (−7,  1.75)  heading west
    Ego crosses Target's lane (x=−1.75) mid-arc at approximately (−1.75, 0)
  Segment 3 — straight exit, west arm:
    (−7, 1.75) → (−67, 1.75)

Target path
-----------
  Straight south: (−1.75, 63) → (−1.75, −50)   speed 30 km/h

Collision timing (approximate, at constant speeds)
---------------------------------------------------
  Ego  : y=−20 → junction entry at 2.778 m/s = 4.68 s straight
         + arc length to θ=53° (crossing x=−1.75) = 8.11 m / 2.778 = 2.92 s
         → reaches (−1.75, 0) at t ≈ 7.6 s
  Target: y=63 → y=0 at 8.333 m/s = 7.56 s
         → arrives at (−1.75, 0) at t ≈ 7.56 s   ✓ near-simultaneous

NCAP parameters used
--------------------
  Ego speed          : 10 km/h (2.778 m/s)
  Target speed       : 30 km/h (8.333 m/s)
  Road arm length    : 60 m (truncated from NCAP 250 m for practical sim)
  Turn radius (Ego)  : 8.75 m  (≈ 2.5 × lane_width)

Collision safety
----------------
  Physics collision between Ego and Target is enabled via MuJoCo's contact
  solver.  An AABB overlap check records the first collision time/speed.

Usage
-----
  mjpython scenarios/NCAP_AEB_C2C_CCFtap/index.py
  mjpython scenarios/NCAP_AEB_C2C_CCFtap/index.py --viewer
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from pure_pursuit import (PurePursuit, straight_segment, arc_segment,
                          make_path, heading_from_xmat, draw_path_geoms)
from props import add_junction_props

SCENE_PATH  = ROOT_DIR / "scenes" / "ncap_junction.xml"
MODEL_PATH  = ROOT_DIR / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "ncap_ccftap.csv"

# ── NCAP parameters ────────────────────────────────────────────────────────────
EGO_SPEED_KPH    = 10.0
EGO_SPEED_MS     = EGO_SPEED_KPH / 3.6          # ≈ 2.778 m/s
TARGET_SPEED_KPH = 30.0
TARGET_SPEED_MS  = TARGET_SPEED_KPH / 3.6        # ≈ 8.333 m/s

# Spawn positions (must match ncap_junction.xml frame positions)
EGO_START_X    =  1.75
EGO_START_Y    = -20.0
TARGET_START_X = -1.75
TARGET_START_Y =  63.0

# Arc geometry: Ego left turn
# Centre at (-7, -7), R=8.75 m → entry (1.75, -7), exit (-7, 1.75)
ARC_CX, ARC_CY, ARC_R = -7.0, -7.0, 8.75

JUNCTION_HALF = 7.0  # junction box extends ±7 m

WHEELBASE_M    = 3.128
WHEEL_RADIUS_M = 0.346
LOOKAHEAD_EGO_M    = 4.0   # tight lookahead for the arc
LOOKAHEAD_TARGET_M = 8.0
TEST_DURATION_S    = 20.0

EGO_HALF_LEN    = 4.973 / 2
EGO_HALF_WID    = 1.941 / 2
TARGET_HALF_LEN = EGO_HALF_LEN
TARGET_HALF_WID = EGO_HALF_WID

COLOR_EGO    = [0.3, 0.5, 1.0, 0.9]
COLOR_TARGET = [1.0, 0.3, 0.2, 0.9]

# Approximate time to collision (for reporting)
_straight_dist = abs(EGO_START_Y) - JUNCTION_HALF          # 20 - 7 = 13 m
_arc_angle_to_coll = math.acos((TARGET_START_X - ARC_CX) / ARC_R)  # θ when x = -1.75
_arc_len_to_coll   = ARC_R * _arc_angle_to_coll
APPROX_TTC_S = (_straight_dist + _arc_len_to_coll) / EGO_SPEED_MS

# ── Reference paths ────────────────────────────────────────────────────────────

PATH_EGO = make_path(
    straight_segment(EGO_START_X, EGO_START_Y, EGO_START_X, -JUNCTION_HALF),
    arc_segment(ARC_CX, ARC_CY, ARC_R, 0.0, math.pi / 2),     # CCW: 0 → π/2
    straight_segment(-JUNCTION_HALF, 1.75, -67.0, 1.75),       # exit west arm
)

PATH_TARGET = straight_segment(TARGET_START_X, TARGET_START_Y,
                                TARGET_START_X, -50.0)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> mujoco.MjModel:
    scene_spec  = mujoco.MjSpec.from_file(str(SCENE_PATH))
    spec_ego    = mujoco.MjSpec.from_file(str(MODEL_PATH))
    spec_target = mujoco.MjSpec.from_file(str(MODEL_PATH))

    add_junction_props(scene_spec, seed=3)

    scene_spec.attach(spec_ego,    frame="spawn_ego",    prefix="ego-")
    scene_spec.attach(spec_target, frame="spawn_target", prefix="target-")

    m = scene_spec.compile()

    # Floor collides with all vehicles (contype=3 matches default vehicle contype=1).
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        m.geom_contype[floor_id]     = 3
        m.geom_conaffinity[floor_id] = 3

    return m


# ── Vehicle helpers ────────────────────────────────────────────────────────────

def _vehicle_ids(m: mujoco.MjModel, prefix: str) -> dict:
    def _aid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    def _sid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   n)
    def _bid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     n)
    def _jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,    n)
    return {
        "steering": _aid(f"{prefix}steering"),
        "throttle": _aid(f"{prefix}throttle"),
        "brake":    _aid(f"{prefix}brake"),
        "vel":      _sid(f"{prefix}velocimeter"),
        "acc":      _sid(f"{prefix}accelerometer"),
        "chassis":  _bid(f"{prefix}chassis"),
        "root":     _jid(f"{prefix}root"),
        "fl_spin":  _jid(f"{prefix}fl_spin"),
        "fr_spin":  _jid(f"{prefix}fr_spin"),
        "rl_spin":  _jid(f"{prefix}rl_spin"),
        "rr_spin":  _jid(f"{prefix}rr_spin"),
        "fl_steer": _jid(f"{prefix}wheel_fl_steering"),
    }


def _init_speed(m, d, ids, speed_ms, direction=+1.0):
    root_dof = int(m.jnt_dofadr[ids["root"]])
    d.qvel[root_dof + 1] = speed_ms * direction
    omega = speed_ms / WHEEL_RADIUS_M
    for jn in ("fl_spin", "fr_spin", "rl_spin", "rr_spin"):
        d.qvel[int(m.jnt_dofadr[ids[jn]])] = omega


def _step_vehicle(m, d, ids, tracker, target_speed):
    chassis = ids["chassis"]
    cx  = float(d.xpos[chassis][0])
    cy  = float(d.xpos[chassis][1])
    hdg = heading_from_xmat(d.xmat, chassis)
    vel_adr = int(m.sensor_adr[ids["vel"]])
    spd = math.sqrt(float(d.sensordata[vel_adr])**2 +
                    float(d.sensordata[vel_adr + 1])**2)

    speed_err = target_speed - spd
    throttle  = max(0.0, min(1.0, 0.15 + 0.6 * speed_err))

    d.ctrl[ids["steering"]] = tracker.compute_ctrl(cx, cy, hdg)
    d.ctrl[ids["throttle"]] = throttle
    d.ctrl[ids["brake"]]    = 0.0


def _check_collision(ego_cx, ego_cy, tgt_cx, tgt_cy):
    """
    AABB overlap for a T-bone: Ego heading ~west, Target heading ~south.
    Use half_len for the dominant axis and half_wid for the lateral axis.
    Conservative: use max(half_len, half_wid) for both directions.
    """
    r = max(EGO_HALF_LEN, EGO_HALF_WID)
    dx = abs(ego_cx - tgt_cx)
    dy = abs(ego_cy - tgt_cy)
    return dx < 2 * r and dy < 2 * r


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel, show_viewer: bool = False,
                   show_labels: bool = True) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    ids_ego    = _vehicle_ids(m, "ego-")
    ids_target = _vehicle_ids(m, "target-")

    _init_speed(m, d, ids_ego,    EGO_SPEED_MS,    +1.0)
    _init_speed(m, d, ids_target, TARGET_SPEED_MS, -1.0)  # facing south

    tracker_ego    = PurePursuit(PATH_EGO,    WHEELBASE_M, LOOKAHEAD_EGO_M)
    tracker_target = PurePursuit(PATH_TARGET, WHEELBASE_M, LOOKAHEAD_TARGET_M)

    times            = []
    ego_xs, ego_ys   = [], []
    target_xs, target_ys = [], []
    ego_speeds       = []
    accel_lons, accel_lats = [], []
    steer_angles     = []
    collision_flags  = []

    collision_time   = None
    collision_speed  = None
    ego_entered_junction = False
    ego_completed_turn   = False

    ego_vel_adr = int(m.sensor_adr[ids_ego["vel"]])
    ego_acc_adr = int(m.sensor_adr[ids_ego["acc"]])
    ego_fl_qpos = int(m.jnt_qposadr[ids_ego["fl_steer"]])

    def _record(tgt_x, tgt_y, collided):
        pos = d.xpos[ids_ego["chassis"]]
        vx  = float(d.sensordata[ego_vel_adr])
        vy  = float(d.sensordata[ego_vel_adr + 1])
        spd = math.sqrt(vx**2 + vy**2)
        times.append(d.time)
        ego_xs.append(float(pos[0]))
        ego_ys.append(float(pos[1]))
        target_xs.append(tgt_x)
        target_ys.append(tgt_y)
        ego_speeds.append(spd)
        accel_lons.append(float(d.sensordata[ego_acc_adr]))
        accel_lats.append(float(d.sensordata[ego_acc_adr + 1]))
        steer_angles.append(float(d.qpos[ego_fl_qpos]))
        collision_flags.append(int(collided))

    def _sim_step():
        if collision_time is None:
            _step_vehicle(m, d, ids_ego,    tracker_ego,    EGO_SPEED_MS)
            _step_vehicle(m, d, ids_target, tracker_target, TARGET_SPEED_MS)
        else:
            for ids_v in (ids_ego, ids_target):
                d.ctrl[ids_v["throttle"]] = 0.0
                d.ctrl[ids_v["brake"]]    = 0.0
                d.ctrl[ids_v["steering"]] = 0.0
        mujoco.mj_step(m, d)
        tgt = d.xpos[ids_target["chassis"]]
        return float(tgt[0]), float(tgt[1])

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                t0 = time.time()
                tgt_x, tgt_y = _sim_step()
                ego_pos  = d.xpos[ids_ego["chassis"]]
                ex, ey   = float(ego_pos[0]), float(ego_pos[1])
                collided = _check_collision(ex, ey, tgt_x, tgt_y)
                _record(tgt_x, tgt_y, collided)
                if collided and collision_time is None:
                    collision_time  = d.time
                    collision_speed = ego_speeds[-1]

                geom_idx = 0
                if show_labels:
                    for ids_v, tracker, path, color, name in [
                        (ids_ego,    tracker_ego,    PATH_EGO,    COLOR_EGO,    "Ego"),
                        (ids_target, tracker_target, PATH_TARGET, COLOR_TARGET, "Target"),
                    ]:
                        cx  = float(d.xpos[ids_v["chassis"]][0])
                        cy  = float(d.xpos[ids_v["chassis"]][1])
                        lp  = tracker.lookahead_point(cx, cy)
                        lbl = d.xpos[ids_v["chassis"]].copy(); lbl[2] += 3.5
                        va  = int(m.sensor_adr[ids_v["vel"]])
                        kph = math.sqrt(float(d.sensordata[va])**2 +
                                        float(d.sensordata[va+1])**2) * 3.6
                        geom_idx = draw_path_geoms(
                            viewer, path, color=color,
                            lookahead_xy=lp, label_pos=lbl,
                            label_text=f"{name} {kph:.0f}km/h",
                            start_idx=geom_idx,
                        )

                    in_jbox = (abs(ex) <= JUNCTION_HALF and abs(ey) <= JUNCTION_HALF)
                    if geom_idx < viewer.user_scn.maxgeom:
                        import mujoco as _mj
                        lbl_pos = d.xpos[ids_ego["chassis"]].copy(); lbl_pos[2] += 6.0
                        g = viewer.user_scn.geoms[geom_idx]
                        _mj.mjv_initGeom(g, _mj.mjtGeom.mjGEOM_LABEL,
                                         np.zeros(3), np.asarray(lbl_pos),
                                         np.eye(3).flatten(),
                                         np.array([1.,1.,0.,1.], dtype=np.float32))
                        g.label = (f"Ego@({ex:.1f},{ey:.1f})  "
                                   f"{'[JUNCTION]' if in_jbox else ''}  "
                                   f"{'*** COLLISION ***' if collided else ''}")
                        geom_idx += 1

                viewer.user_scn.ngeom = geom_idx
                viewer.sync()
                sleep_t = m.opt.timestep - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            tgt_x, tgt_y = _sim_step()
            ego_pos  = d.xpos[ids_ego["chassis"]]
            ex, ey   = float(ego_pos[0]), float(ego_pos[1])
            collided = _check_collision(ex, ey, tgt_x, tgt_y)
            _record(tgt_x, tgt_y, collided)
            if collided and collision_time is None:
                collision_time  = d.time
                collision_speed = ego_speeds[-1]

    return {
        "time_s":             np.array(times),
        "ego_x_m":            np.array(ego_xs),
        "ego_y_m":            np.array(ego_ys),
        "target_x_m":         np.array(target_xs),
        "target_y_m":         np.array(target_ys),
        "speed_ms":           np.array(ego_speeds),
        "accel_lon_ms2":      np.array(accel_lons),
        "accel_lat_ms2":      np.array(accel_lats),
        "steer_angle_rad":    np.array(steer_angles),
        "collision":          np.array(collision_flags),
        "collision_time_s":   collision_time,
        "collision_speed_ms": collision_speed,
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict:
    xs, ys = data["ego_x_m"], data["ego_y_m"]
    speeds = data["speed_ms"]

    # Ego entered the junction box
    in_jbox = (np.abs(xs) <= JUNCTION_HALF) & (np.abs(ys) <= JUNCTION_HALF)
    ego_entered_junction = bool(in_jbox.any())

    # Ego crossed into negative x (began left turn past the centre line)
    ego_crossed_centreline = bool(np.any(xs < -0.5))

    # Ego exited into west arm (x < -JUNCTION_HALF)
    ego_in_west = xs < -JUNCTION_HALF
    ego_reached_west = bool(ego_in_west.any())

    return {
        "mean_ego_speed_kph":    float(np.mean(speeds)) * 3.6,
        "peak_ego_speed_kph":    float(np.max(speeds))  * 3.6,
        "ego_entered_junction":  ego_entered_junction,
        "ego_crossed_centreline": ego_crossed_centreline,
        "ego_reached_west":      ego_reached_west,
        "collision_detected":    data["collision_time_s"] is not None,
        "collision_time_s":      data["collision_time_s"] or float("nan"),
        "collision_speed_kph":   (data["collision_speed_ms"] * 3.6
                                  if data["collision_speed_ms"] is not None
                                  else float("nan")),
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    ok = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")
    print(f"  [INFO] Mean Ego speed:       {metrics['mean_ego_speed_kph']:.1f} km/h")
    print(f"  [INFO] Peak Ego speed:       {metrics['peak_ego_speed_kph']:.1f} km/h")
    print(f"  [INFO] Approx. expected TTC: {APPROX_TTC_S:.1f} s")
    if not math.isnan(metrics["collision_time_s"]):
        print(f"  [INFO] Collision time:       {metrics['collision_time_s']:.3f} s")
        print(f"  [INFO] Ego speed at coll.:   {metrics['collision_speed_kph']:.1f} km/h")

    ok1 = EGO_SPEED_KPH * 0.7 <= metrics["mean_ego_speed_kph"] <= EGO_SPEED_KPH * 1.3
    ok  = ok and ok1
    print(f"\n  [{'PASS' if ok1 else 'FAIL'}] Ego maintains ~{EGO_SPEED_KPH:.0f} km/h "
          f"(mean = {metrics['mean_ego_speed_kph']:.1f} km/h)")

    ok2 = metrics["ego_entered_junction"]
    ok  = ok and ok2
    print(f"\n  [{'PASS' if ok2 else 'FAIL'}] Ego entered junction box (±{JUNCTION_HALF} m)")

    ok3 = metrics["ego_crossed_centreline"]
    ok  = ok and ok3
    print(f"\n  [{'PASS' if ok3 else 'FAIL'}] Ego crossed road centreline (left turn commenced)")

    ok4 = metrics["collision_detected"]
    ok  = ok and ok4
    print(f"\n  [{'PASS' if ok4 else 'FAIL'}] Collision detected "
          f"(expected without AEB — "
          f"t={metrics['collision_time_s']:.2f} s, "
          f"v={metrics['collision_speed_kph']:.1f} km/h)")

    return ok


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([
        data["time_s"],
        data["ego_x_m"],     data["ego_y_m"],
        data["target_x_m"],  data["target_y_m"],
        data["speed_ms"],    data["speed_ms"] * 3.6,
        data["accel_lon_ms2"], data["accel_lat_ms2"],
        data["steer_angle_rad"],
        data["collision"],
    ])
    np.savetxt(CSV_PATH, rows, delimiter=",",
               header="time_s,ego_x_m,ego_y_m,target_x_m,target_y_m,"
                      "ego_speed_ms,ego_speed_kph,"
                      "accel_lon_ms2,accel_lat_ms2,steer_angle_rad,collision",
               comments="")
    print(f"\n  CSV saved → {CSV_PATH}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NCAP AEB C2C CCFtap — Ego turns left, Target crosses straight")
    parser.add_argument("--viewer",   action="store_true", default=False)
    parser.add_argument("--no-label", action="store_true", default=False,
                        help="Hide speed/position overlays in the viewer")
    args = parser.parse_args()

    print("=" * 70)
    print("  NCAP AEB C2C CCFtap — Car-to-Car Front Turning Across Path")
    print("=" * 70)
    print(f"  Scene:           {SCENE_PATH}")
    print(f"  Ego speed:       {EGO_SPEED_KPH} km/h  (northbound → left turn)")
    print(f"  Target speed:    {TARGET_SPEED_KPH} km/h  (southbound, straight)")
    print(f"  Ego start:       ({EGO_START_X}, {EGO_START_Y})")
    print(f"  Target start:    ({TARGET_START_X}, {TARGET_START_Y})")
    print(f"  Arc:             centre ({ARC_CX}, {ARC_CY}), R={ARC_R} m")
    print(f"  Approx. TTC:     {APPROX_TTC_S:.1f} s")
    print(f"  Path points:     Ego={len(PATH_EGO)}, Target={len(PATH_TARGET)}")
    print(f"  Viewer:          {'on' if args.viewer else 'off (pass --viewer)'}")

    for p, label in [(SCENE_PATH, "scene"), (MODEL_PATH, "model")]:
        if not p.exists():
            print(f"\n  ERROR: {label} not found: {p}")
            sys.exit(1)

    m = build_model()
    print("\n  Running simulation...")
    data    = run_simulation(m, show_viewer=args.viewer,
                             show_labels=not args.no_label)
    metrics = extract_metrics(data)
    ok      = print_report(metrics)
    save_csv(data)

    print("\n" + "=" * 70)
    print(f"  {'PASS' if ok else 'FAIL'} — NCAP AEB C2C CCFtap")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
