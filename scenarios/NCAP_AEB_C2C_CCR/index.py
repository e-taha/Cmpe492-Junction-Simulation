"""
NCAP AEB C2C CCR — Car-to-Car Rear (2023)
==========================================
Source scenario: NCAP_scenarios/NCAP_AEB_C2C_CCR_2023.xosc

Covers three sub-scenarios via --mode flag:

  CCRs  (stationary)  : GVT is stationary. Ego rear-ends it.
  CCRm  (moving)      : GVT at constant speed < Ego. Ego closes the gap.
  CCRb  (braking)     : GVT decelerates from GVT_SPEED_MS to 0 after a delay.

Both vehicles are in the northbound lane (x=+1.75 m), same direction.
GVT starts ahead of Ego with Ego_initTimeHeadway = 5 s separation.

NCAP default parameters
-----------------------
  Ego_speed_kph          : 20  (CCRs default; can override via --ego-speed)
  GVT_init_speed_kph     : 0   (CCRs) | 10  (CCRm/CCRb)
  Ego_initTimeHeadway    : 5 s
  GVT_deceleration       : 2 m/s² (CCRb only)
  GVT_braking_delay      : 3 s   (CCRb only)

Collision safety
----------------
  GVT geoms: contype=2 / conaffinity=2  — no physics collision with Ego.
  Floor:     contype=3 / conaffinity=3
  Ego geoms: contype=1 / conaffinity=1
  Virtual collision via bounding-box overlap check.

Usage
-----
  mjpython scenarios/NCAP_AEB_C2C_CCR/index.py
  mjpython scenarios/NCAP_AEB_C2C_CCR/index.py --mode ccrm
  mjpython scenarios/NCAP_AEB_C2C_CCR/index.py --mode ccrb
  mjpython scenarios/NCAP_AEB_C2C_CCR/index.py --viewer
  mjpython scenarios/NCAP_AEB_C2C_CCR/index.py --ego-speed 30 --mode ccrs
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
from pure_pursuit import (PurePursuit, straight_segment,
                          heading_from_xmat, draw_path_geoms)

SCENE_PATH  = ROOT_DIR / "scenes" / "ncap_straight_road_ccr.xml"
MODEL_PATH  = ROOT_DIR / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"

# ── NCAP parameters ────────────────────────────────────────────────────────────
EGO_INIT_TIMEHEADWAY_S = 5.0
GVT_DECEL_MS2          = 2.0    # CCRb braking rate (m/s²)
GVT_BRAKING_DELAY_S    = 3.0    # CCRb: delay before GVT brakes

WHEELBASE_M   = 3.128
WHEEL_RADIUS_M = 0.346
LOOKAHEAD_M   = 6.0
TEST_DURATION_S = 25.0

LANE_X        = 1.75
EGO_HALF_LEN  = 4.973 / 2
EGO_HALF_WID  = 1.941 / 2
GVT_HALF_LEN  = EGO_HALF_LEN
GVT_HALF_WID  = EGO_HALF_WID

COLOR_EGO = [0.3, 0.5, 1.0, 0.9]
COLOR_GVT = [1.0, 0.3, 0.2, 0.9]


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> mujoco.MjModel:
    scene_spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    spec_ego   = mujoco.MjSpec.from_file(str(MODEL_PATH))
    spec_gvt   = mujoco.MjSpec.from_file(str(MODEL_PATH))

    scene_spec.attach(spec_ego, frame="spawn_ego", prefix="ego-")
    scene_spec.attach(spec_gvt, frame="spawn_gvt", prefix="gvt-")

    m = scene_spec.compile()

    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        m.geom_contype[floor_id]     = 3
        m.geom_conaffinity[floor_id] = 3

    for i in range(m.ngeom):
        body_id   = int(m.geom_bodyid[i])
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith("gvt-"):
            m.geom_contype[i]     = 2
            m.geom_conaffinity[i] = 2

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


def _brake_vehicle(m, d, ids, decel_ms2):
    """Apply braking to decelerate at the requested rate (approximation)."""
    vel_adr = int(m.sensor_adr[ids["vel"]])
    spd = math.sqrt(float(d.sensordata[vel_adr])**2 +
                    float(d.sensordata[vel_adr + 1])**2)
    if spd < 0.1:
        d.ctrl[ids["throttle"]] = 0.0
        d.ctrl[ids["brake"]]    = 0.0
        return
    brake_ctrl = max(0.0, min(1.0, decel_ms2 / 4.0))
    d.ctrl[ids["throttle"]] = 0.0
    d.ctrl[ids["brake"]]    = brake_ctrl


def _check_collision(ego_x, ego_y, gvt_x, gvt_y):
    # Ego front past GVT rear
    lon = (ego_y + EGO_HALF_LEN) > (gvt_y - GVT_HALF_LEN)
    lat = abs(ego_x - gvt_x) < (EGO_HALF_WID + GVT_HALF_WID)
    return lon and lat


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel, ego_speed_ms: float,
                   gvt_init_speed_ms: float, mode: str,
                   show_viewer: bool = False) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    ids_ego = _vehicle_ids(m, "ego-")
    ids_gvt = _vehicle_ids(m, "gvt-")

    # Compute actual initial separation from NCAP headway
    init_sep_m = EGO_INIT_TIMEHEADWAY_S * ego_speed_ms

    # Adjust GVT spawn from the scene default (y=28).
    # GVT spawn frame is at y=28; adjust qpos if needed.
    # GVT faces north (local +x = world +y for euler "0 0 90").
    gvt_root_qpos = int(m.jnt_qposadr[ids_gvt["root"]])
    scene_gvt_y = 28.0  # matches spawn_gvt pos in ncap_straight_road_ccr.xml
    gvt_y_offset = init_sep_m - scene_gvt_y
    d.qpos[gvt_root_qpos + 0] = gvt_y_offset  # local x = world y for north-facing

    _init_speed(m, d, ids_ego, ego_speed_ms,      +1.0)
    _init_speed(m, d, ids_gvt, gvt_init_speed_ms, +1.0)
    mujoco.mj_forward(m, d)  # propagate adjusted position

    path_ego = straight_segment(LANE_X, -5.0, LANE_X, 300.0)
    path_gvt = straight_segment(LANE_X,  init_sep_m - 10.0, LANE_X, 300.0)

    tracker_ego = PurePursuit(path_ego, WHEELBASE_M, LOOKAHEAD_M)
    tracker_gvt = PurePursuit(path_gvt, WHEELBASE_M, LOOKAHEAD_M)

    times            = []
    ego_ys, gvt_ys   = [], []
    ego_speeds, gvt_speeds = [], []
    accel_lons, accel_lats = [], []
    steer_angles     = []
    separations      = []
    collision_flags  = []
    gvt_ctrl_mode    = []  # 'const' | 'brake' | 'stopped'

    collision_time   = None
    collision_speed  = None

    ego_vel_adr = int(m.sensor_adr[ids_ego["vel"]])
    ego_acc_adr = int(m.sensor_adr[ids_ego["acc"]])
    ego_fl_qpos = int(m.jnt_qposadr[ids_ego["fl_steer"]])
    gvt_vel_adr = int(m.sensor_adr[ids_gvt["vel"]])

    gvt_braking_started = False
    gvt_headway_established = False
    gvt_headway_time = None

    def _record(gvt_y, gvt_spd, collided, ctrl_label):
        pos = d.xpos[ids_ego["chassis"]]
        vx  = float(d.sensordata[ego_vel_adr])
        vy  = float(d.sensordata[ego_vel_adr + 1])
        spd = math.sqrt(vx**2 + vy**2)
        times.append(d.time)
        ego_ys.append(float(pos[1]))
        gvt_ys.append(gvt_y)
        ego_speeds.append(spd)
        gvt_speeds.append(gvt_spd)
        accel_lons.append(float(d.sensordata[ego_acc_adr]))
        accel_lats.append(float(d.sensordata[ego_acc_adr + 1]))
        steer_angles.append(float(d.qpos[ego_fl_qpos]))
        separations.append(max(0.0, gvt_y - float(pos[1]) - EGO_HALF_LEN - GVT_HALF_LEN))
        collision_flags.append(int(collided))
        gvt_ctrl_mode.append(ctrl_label)

    def _get_gvt_speed():
        return math.sqrt(float(d.sensordata[gvt_vel_adr])**2 +
                         float(d.sensordata[gvt_vel_adr + 1])**2)

    def _apply_gvt(ctrl_label_ref):
        nonlocal gvt_braking_started, gvt_headway_established, gvt_headway_time
        ctrl_label = "const"
        if mode == "ccrs":
            # GVT stationary — no actuator input
            d.ctrl[ids_gvt["throttle"]] = 0.0
            d.ctrl[ids_gvt["brake"]]    = 0.0
            d.ctrl[ids_gvt["steering"]] = 0.0
            ctrl_label = "stopped"
        elif mode == "ccrm":
            _step_vehicle(m, d, ids_gvt, tracker_gvt, gvt_init_speed_ms)
            ctrl_label = "const"
        elif mode == "ccrb":
            # Once headway is established (t > 0.5 s), wait braking_delay, then brake
            if not gvt_headway_established and d.time > 0.5:
                gvt_headway_established = True
                gvt_headway_time = d.time
            if gvt_headway_established:
                elapsed = d.time - gvt_headway_time
                if elapsed < GVT_BRAKING_DELAY_S:
                    _step_vehicle(m, d, ids_gvt, tracker_gvt, gvt_init_speed_ms)
                    ctrl_label = "const"
                else:
                    _brake_vehicle(m, d, ids_gvt, GVT_DECEL_MS2)
                    ctrl_label = "brake"
            else:
                _step_vehicle(m, d, ids_gvt, tracker_gvt, gvt_init_speed_ms)
                ctrl_label = "const"
        ctrl_label_ref[0] = ctrl_label

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                t0 = time.time()
                _step_vehicle(m, d, ids_ego, tracker_ego, ego_speed_ms)
                ctrl_ref = ["const"]
                _apply_gvt(ctrl_ref)
                mujoco.mj_step(m, d)

                gvt_pos = d.xpos[ids_gvt["chassis"]]
                gvt_y   = float(gvt_pos[1])
                gvt_x   = float(gvt_pos[0])
                gvt_spd = _get_gvt_speed()
                ego_pos = d.xpos[ids_ego["chassis"]]
                ex, ey  = float(ego_pos[0]), float(ego_pos[1])
                collided = _check_collision(ex, ey, gvt_x, gvt_y)
                _record(gvt_y, gvt_spd, collided, ctrl_ref[0])
                if collided and collision_time is None:
                    collision_time  = d.time
                    collision_speed = ego_speeds[-1]

                geom_idx = 0
                for ids_v, tracker, path, color, name in [
                    (ids_ego, tracker_ego, path_ego, COLOR_EGO, "Ego"),
                    (ids_gvt, tracker_gvt, path_gvt, COLOR_GVT, "GVT"),
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
                        label_text=f"{name} {kph:.0f}km/h [{ctrl_ref[0]}]",
                        start_idx=geom_idx,
                    )

                sep = max(0.0, gvt_y - ey - EGO_HALF_LEN - GVT_HALF_LEN)
                if geom_idx < viewer.user_scn.maxgeom:
                    import mujoco as _mj
                    lbl_pos = d.xpos[ids_ego["chassis"]].copy(); lbl_pos[2] += 6.0
                    g = viewer.user_scn.geoms[geom_idx]
                    _mj.mjv_initGeom(g, _mj.mjtGeom.mjGEOM_LABEL,
                                     np.zeros(3), np.asarray(lbl_pos),
                                     np.eye(3).flatten(),
                                     np.array([1.,1.,0.,1.], dtype=np.float32))
                    g.label = (f"sep={sep:.1f}m  mode={mode.upper()}  "
                               f"{'*** COLLISION ***' if collided else ''}")
                    geom_idx += 1

                viewer.user_scn.ngeom = geom_idx
                viewer.sync()
                sleep_t = m.opt.timestep - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            _step_vehicle(m, d, ids_ego, tracker_ego, ego_speed_ms)
            ctrl_ref = ["const"]
            _apply_gvt(ctrl_ref)
            mujoco.mj_step(m, d)

            gvt_pos  = d.xpos[ids_gvt["chassis"]]
            gvt_y    = float(gvt_pos[1])
            gvt_x    = float(gvt_pos[0])
            gvt_spd  = _get_gvt_speed()
            ego_pos  = d.xpos[ids_ego["chassis"]]
            ex, ey   = float(ego_pos[0]), float(ego_pos[1])
            collided = _check_collision(ex, ey, gvt_x, gvt_y)
            _record(gvt_y, gvt_spd, collided, ctrl_ref[0])
            if collided and collision_time is None:
                collision_time  = d.time
                collision_speed = ego_speeds[-1]

    return {
        "time_s":             np.array(times),
        "ego_y_m":            np.array(ego_ys),
        "gvt_y_m":            np.array(gvt_ys),
        "ego_speed_ms":       np.array(ego_speeds),
        "gvt_speed_ms":       np.array(gvt_speeds),
        "accel_lon_ms2":      np.array(accel_lons),
        "accel_lat_ms2":      np.array(accel_lats),
        "steer_angle_rad":    np.array(steer_angles),
        "separation_m":       np.array(separations),
        "collision":          np.array(collision_flags),
        "gvt_ctrl":           gvt_ctrl_mode,
        "collision_time_s":   collision_time,
        "collision_speed_ms": collision_speed,
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def extract_metrics(data: dict, ego_speed_ms: float) -> dict:
    speeds = data["ego_speed_ms"]
    return {
        "mean_ego_speed_kph":   float(np.mean(speeds)) * 3.6,
        "peak_ego_speed_kph":   float(np.max(speeds))  * 3.6,
        "target_speed_kph":     ego_speed_ms * 3.6,
        "collision_detected":   data["collision_time_s"] is not None,
        "collision_time_s":     data["collision_time_s"] or float("nan"),
        "collision_speed_kph":  (data["collision_speed_ms"] * 3.6
                                 if data["collision_speed_ms"] is not None
                                 else float("nan")),
        "min_separation_m":     float(np.min(data["separation_m"])),
        "init_separation_m":    float(data["separation_m"][0]) if len(data["separation_m"]) else float("nan"),
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(metrics: dict, mode: str) -> bool:
    ok = True
    tgt = metrics["target_speed_kph"]
    print("\n── Metrics ────────────────────────────────────────────────────────────")
    print(f"  [INFO] Mode:                {mode.upper()}")
    print(f"  [INFO] Initial separation:  {metrics['init_separation_m']:.1f} m (freespace)")
    print(f"  [INFO] Mean Ego speed:      {metrics['mean_ego_speed_kph']:.1f} km/h")
    print(f"  [INFO] Peak Ego speed:      {metrics['peak_ego_speed_kph']:.1f} km/h")
    if not math.isnan(metrics["collision_time_s"]):
        print(f"  [INFO] Collision time:      {metrics['collision_time_s']:.3f} s")
        print(f"  [INFO] Ego speed at coll.:  {metrics['collision_speed_kph']:.1f} km/h")

    ok1 = tgt * 0.85 <= metrics["mean_ego_speed_kph"] <= tgt * 1.15
    ok  = ok and ok1
    print(f"\n  [{'PASS' if ok1 else 'FAIL'}] Ego maintains ~{tgt:.0f} km/h "
          f"(mean = {metrics['mean_ego_speed_kph']:.1f} km/h)")

    ok2 = metrics["collision_detected"]
    ok  = ok and ok2
    print(f"\n  [{'PASS' if ok2 else 'FAIL'}] Collision detected "
          f"(expected without AEB) — "
          f"t={metrics['collision_time_s']:.2f} s, "
          f"v={metrics['collision_speed_kph']:.1f} km/h")

    return ok


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(data: dict, mode: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"ncap_ccr_{mode}.csv"
    rows = np.column_stack([
        data["time_s"],
        data["ego_y_m"],    data["gvt_y_m"],
        data["ego_speed_ms"], data["ego_speed_ms"] * 3.6,
        data["gvt_speed_ms"], data["gvt_speed_ms"] * 3.6,
        data["accel_lon_ms2"], data["accel_lat_ms2"],
        data["steer_angle_rad"],
        data["separation_m"],  data["collision"],
    ])
    np.savetxt(csv_path, rows, delimiter=",",
               header="time_s,ego_y_m,gvt_y_m,"
                      "ego_speed_ms,ego_speed_kph,"
                      "gvt_speed_ms,gvt_speed_kph,"
                      "accel_lon_ms2,accel_lat_ms2,steer_angle_rad,"
                      "freespace_sep_m,collision",
               comments="")
    print(f"\n  CSV saved → {csv_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NCAP AEB C2C CCR — rear-end scenario (CCRs / CCRm / CCRb)")
    parser.add_argument("--mode",      choices=["ccrs", "ccrm", "ccrb"],
                        default="ccrs", help="CCR sub-scenario (default: ccrs)")
    parser.add_argument("--ego-speed", type=float, default=None,
                        help="Ego speed in km/h (default: 20 for CCRs, 30 for CCRm/CCRb)")
    parser.add_argument("--viewer",    action="store_true", default=False)
    args = parser.parse_args()

    mode = args.mode

    if args.ego_speed is not None:
        ego_speed_kph = args.ego_speed
    else:
        ego_speed_kph = 20.0 if mode == "ccrs" else 30.0

    ego_speed_ms = ego_speed_kph / 3.6

    if mode == "ccrs":
        gvt_init_speed_ms = 0.0
        gvt_label = "stationary"
    elif mode == "ccrm":
        gvt_init_speed_ms = ego_speed_ms / 2.0
        gvt_label = f"constant {gvt_init_speed_ms*3.6:.0f} km/h"
    else:  # ccrb
        gvt_init_speed_ms = ego_speed_ms / 3.0
        gvt_label = f"{gvt_init_speed_ms*3.6:.0f} km/h → brakes"

    init_sep_m = EGO_INIT_TIMEHEADWAY_S * ego_speed_ms

    print("=" * 70)
    print(f"  NCAP AEB C2C {mode.upper()} — Car-to-Car Rear")
    print("=" * 70)
    print(f"  Scene:           {SCENE_PATH}")
    print(f"  Ego speed:       {ego_speed_kph:.0f} km/h")
    print(f"  GVT speed:       {gvt_label}")
    print(f"  Initial sep.:    {init_sep_m:.1f} m  (headway={EGO_INIT_TIMEHEADWAY_S} s)")
    if mode == "ccrb":
        print(f"  Braking delay:   {GVT_BRAKING_DELAY_S} s  "
              f"decel={GVT_DECEL_MS2} m/s²")
    print(f"  Viewer:          {'on' if args.viewer else 'off (pass --viewer)'}")

    for p, label in [(SCENE_PATH, "scene"), (MODEL_PATH, "model")]:
        if not p.exists():
            print(f"\n  ERROR: {label} not found: {p}")
            sys.exit(1)

    m = build_model()
    print("\n  Running simulation...")
    data    = run_simulation(m, ego_speed_ms, gvt_init_speed_ms, mode,
                             show_viewer=args.viewer)
    metrics = extract_metrics(data, ego_speed_ms)
    ok      = print_report(metrics, mode)
    save_csv(data, mode)

    print("\n" + "=" * 70)
    print(f"  {'PASS' if ok else 'FAIL'} — NCAP AEB C2C {mode.upper()}")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
