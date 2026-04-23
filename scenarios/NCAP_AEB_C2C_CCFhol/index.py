"""
NCAP AEB C2C CCFhol — Car-to-Car Front Head-On Lane-Change Overlap (2023)
=========================================================================
Source scenario: NCAP_scenarios/NCAP_AEB_C2C_CCFhol_2023.xosc

Scenario overview
-----------------
  Ego    : northbound at 70 km/h in the right (northbound) lane (x = +1.75 m),
           constant speed, no AEB.

  GVT    : (Global Vehicle Target) southbound at 70 km/h, initially in the left
           (southbound) lane (x = -1.75 m).  After the synchronisation delay the
           GVT performs a smooth cosine lane change over 60 m longitudinal
           distance into the Ego's lane (x = +1.75 m), ending with TTC ≈ 1.5 s.

  SOV    : (Screen Object Vehicle / obstruction) southbound at 70 km/h in the
           left lane, 19.4 m freespace ahead of GVT — it never changes lane.

NCAP parameters (from .xosc)
-----------------------------
  Ego speed              : 70 km/h  (= 19.444 m/s)
  GVT speed              : = Ego speed
  Lane change distance L : 60 m
  Lane change TTC F      : 1.5 s  (TTC when lane change ends)
  Following distance F   : 19.4 m freespace (SOV → GVT gap)
  Lateral offset O       : 3.5 m  (= 2 × half-lane-width)

Derived timing (closing speed = 2 × 19.444 = 38.888 m/s)
----------------------------------------------------------
  Lane change duration   : 60 / 19.444 ≈ 3.086 s
  Lane change start TTC  : 3.086 + 1.5 = 4.586 s before collision
  Gap at LC start        : 4.586 × 38.888 ≈ 178.3 m
  With GVT_START_Y=250, EGO_START_Y=0:
    LC start time        : (250 − 178.3) / 38.888 ≈ 1.844 s

Metrics reported
----------------
  - GVT lane change completed into Ego lane
  - TTC at lane change end (should be ≈ 1.5 s)
  - Collision detected (expected without AEB)
  - Ego speed at collision
  - Time of collision

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
from pure_pursuit import (PurePursuit, straight_segment,
                          heading_from_xmat, draw_overlay)

# ── File paths ─────────────────────────────────────────────────────────────────
SCENE_PATH  = ROOT_DIR / "scenes" / "ncap_straight_road.xml"
MODEL_PATH  = ROOT_DIR / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "ncap_ccfhol.csv"

# ── NCAP scenario parameters ───────────────────────────────────────────────────
EGO_SPEED_KPH       = 70.0
EGO_SPEED_MS        = EGO_SPEED_KPH / 3.6           # ≈ 19.444 m/s
GVT_SPEED_MS        = EGO_SPEED_MS                   # GVT = Ego speed

EGO_START_Y         = 0.0                            # m (northbound lane)
GVT_START_Y         = 250.0                          # m (southbound lane, far north)
SOV_FREESPACE_M     = 19.4                           # m (freespace GVT→SOV)
GVT_HALF_LENGTH     = 4.023 / 2                      # m
SOV_HALF_LENGTH     = 5.0 / 2                        # m (obstruction vehicle)
SOV_CENTER_OFFSET   = SOV_FREESPACE_M + GVT_HALF_LENGTH + SOV_HALF_LENGTH  # ≈ 24.5 m

LANE_EGO_X          = 1.75                           # m (northbound lane centre)
LANE_GVT_X          = -1.75                          # m (southbound lane centre)
LANE_CHANGE_DIST_M  = 60.0                           # m (longitudinal LC distance)
LANE_CHANGE_TTC_S   = 1.5                            # s (TTC when LC ends)

# ── Derived timing ─────────────────────────────────────────────────────────────
CLOSING_SPEED_MS    = EGO_SPEED_MS + GVT_SPEED_MS   # ≈ 38.888 m/s
LC_DURATION_S       = LANE_CHANGE_DIST_M / GVT_SPEED_MS  # ≈ 3.086 s
LC_START_TTC_S      = LC_DURATION_S + LANE_CHANGE_TTC_S  # ≈ 4.586 s
INITIAL_SEP_M       = GVT_START_Y - EGO_START_Y     # 250 m
TIME_TO_COLLISION_S = INITIAL_SEP_M / CLOSING_SPEED_MS  # ≈ 6.43 s
LC_START_TIME_S     = TIME_TO_COLLISION_S - LC_START_TTC_S  # ≈ 1.844 s

# ── Collision geometry ─────────────────────────────────────────────────────────
EGO_HALF_LENGTH     = 4.973 / 2                      # m (Edgar vehicle)
EGO_HALF_WIDTH      = 1.941 / 2                      # m
GVT_HALF_WIDTH      = 1.941 / 2                      # m
COLLISION_THRESHOLD = EGO_HALF_LENGTH + GVT_HALF_LENGTH  # front-to-front ≈ 4.48 m

# ── Vehicle constants ──────────────────────────────────────────────────────────
WHEELBASE_M         = 3.128
LOOKAHEAD_M         = 8.0    # m — long lookahead for straight-line driving
TEST_DURATION_S     = 12.0   # s — enough to see collision or pass-by

# ── Reference path for Ego ─────────────────────────────────────────────────────
PATH_EGO = straight_segment(LANE_EGO_X, -5.0, LANE_EGO_X, 350.0)


# ── GVT / SOV kinematic trajectory ────────────────────────────────────────────

def _gvt_state(t: float) -> tuple:
    """
    Return (x, y, quat) for the GVT mocap body at simulation time t.

    GVT travels south at GVT_SPEED_MS.  At t = LC_START_TIME_S it begins a
    cosine-interpolated lane change from x=LANE_GVT_X to x=LANE_EGO_X over
    LC_DURATION_S seconds.  The instantaneous heading is computed from the
    velocity vector so the body tilts naturally during the manoeuvre.
    """
    gvt_y = GVT_START_Y - GVT_SPEED_MS * t

    if t < LC_START_TIME_S:
        gvt_x  = LANE_GVT_X
        vx     = 0.0
    elif t < LC_START_TIME_S + LC_DURATION_S:
        progress = (t - LC_START_TIME_S) / LC_DURATION_S        # 0 → 1
        alpha    = (1.0 - math.cos(math.pi * progress)) / 2.0   # smooth S-curve
        gvt_x    = LANE_GVT_X + alpha * (LANE_EGO_X - LANE_GVT_X)
        # lateral velocity from derivative of cosine interpolation
        vx = (LANE_EGO_X - LANE_GVT_X) * (math.pi / (2.0 * LC_DURATION_S)) \
             * math.sin(math.pi * progress)
    else:
        gvt_x = LANE_EGO_X
        vx    = 0.0

    vy = -GVT_SPEED_MS  # always southward

    # heading = direction of motion; vehicle forward is local +x
    heading  = math.atan2(vy, vx) if (vx != 0.0 or vy != 0.0) else -math.pi / 2
    q_w = math.cos(heading / 2.0)
    q_z = math.sin(heading / 2.0)
    return gvt_x, gvt_y, (q_w, 0.0, 0.0, q_z)


def _sov_state(gvt_x: float, gvt_y: float) -> tuple:
    """
    SOV stays in the southbound lane (x = LANE_GVT_X), 24.5 m ahead of GVT
    (centre-to-centre in GVT's travel direction = lower y).
    """
    sov_y = gvt_y - SOV_CENTER_OFFSET
    # Always facing south
    q_w = math.cos(-math.pi / 4.0)
    q_z = math.sin(-math.pi / 4.0)
    return LANE_GVT_X, sov_y, (q_w, 0.0, 0.0, q_z)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> mujoco.MjModel:
    scene_spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    robot_spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    scene_spec.attach(robot_spec, frame="spawn_ego", prefix="ego-")
    return scene_spec.compile()


# ── Collision detection ────────────────────────────────────────────────────────

def _check_collision(ego_x: float, ego_y: float,
                     gvt_x: float, gvt_y: float) -> bool:
    """
    Simple axis-aligned bounding box overlap test.
    Both vehicles treated as rectangles aligned with the road axes.
    """
    lon_overlap = abs(ego_y - gvt_y) < (EGO_HALF_LENGTH + GVT_HALF_LENGTH)
    lat_overlap = abs(ego_x - gvt_x) < (EGO_HALF_WIDTH  + GVT_HALF_WIDTH)
    return lon_overlap and lat_overlap


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel, show_viewer: bool = False) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    # IDs for the Ego vehicle
    steering_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "ego-steering")
    throttle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "ego-throttle")
    brake_id    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "ego-brake")
    vel_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "ego-velocimeter")
    acc_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "ego-accelerometer")
    chassis_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     "ego-chassis")
    fl_jnt_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,    "ego-wheel_fl_steering")

    vel_adr = int(m.sensor_adr[vel_id])
    acc_adr = int(m.sensor_adr[acc_id])
    fl_qpos = int(m.jnt_qposadr[fl_jnt_id])

    # Mocap IDs for GVT and SOV
    gvt_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gvt")
    sov_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sov")
    gvt_mocap_id = int(m.body_mocapid[gvt_body_id])
    sov_mocap_id = int(m.body_mocapid[sov_body_id])

    tracker = PurePursuit(PATH_EGO, WHEELBASE_M, LOOKAHEAD_M)

    times, xs, ys                    = [], [], []
    speeds, accel_lons, accel_lats   = [], [], []
    steer_angles                     = []
    gvt_xs, gvt_ys                   = [], []
    separations, collision_flags     = [], []

    collision_time  = None
    collision_speed = None

    def _update_mocap(t: float):
        gvt_x, gvt_y, gvt_q = _gvt_state(t)
        sov_x, sov_y, sov_q = _sov_state(gvt_x, gvt_y)

        d.mocap_pos[gvt_mocap_id, 0] = gvt_x
        d.mocap_pos[gvt_mocap_id, 1] = gvt_y
        d.mocap_pos[gvt_mocap_id, 2] = 1.333
        d.mocap_quat[gvt_mocap_id]   = gvt_q

        d.mocap_pos[sov_mocap_id, 0] = sov_x
        d.mocap_pos[sov_mocap_id, 1] = sov_y
        d.mocap_pos[sov_mocap_id, 2] = 1.5
        d.mocap_quat[sov_mocap_id]   = sov_q

        return gvt_x, gvt_y

    def _step_controls(cx: float, cy: float, hdg: float,
                       current_speed: float) -> None:
        # Speed controller: hold EGO_SPEED_MS
        speed_err = EGO_SPEED_MS - current_speed
        throttle  = max(0.0, min(1.0, 0.15 + 0.6 * speed_err))
        d.ctrl[steering_id] = tracker.compute_ctrl(cx, cy, hdg)
        d.ctrl[throttle_id] = throttle
        d.ctrl[brake_id]    = 0.0

    def _record(gvt_x: float, gvt_y: float, collided: bool) -> None:
        vx  = float(d.sensordata[vel_adr])
        vy  = float(d.sensordata[vel_adr + 1])
        pos = d.xpos[chassis_id]
        spd = math.sqrt(vx**2 + vy**2)
        sep = abs(float(pos[1]) - gvt_y)

        times.append(d.time)
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        speeds.append(spd)
        accel_lons.append(float(d.sensordata[acc_adr]))
        accel_lats.append(float(d.sensordata[acc_adr + 1]))
        steer_angles.append(float(d.qpos[fl_qpos]))
        gvt_xs.append(gvt_x)
        gvt_ys.append(gvt_y)
        separations.append(sep)
        collision_flags.append(int(collided))

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                t0 = time.time()

                gvt_x, gvt_y = _update_mocap(d.time)
                pos = d.xpos[chassis_id]
                cx, cy = float(pos[0]), float(pos[1])
                hdg = heading_from_xmat(d.xmat, chassis_id)
                vx  = float(d.sensordata[vel_adr])
                vy  = float(d.sensordata[vel_adr + 1])
                spd = math.sqrt(vx**2 + vy**2)

                _step_controls(cx, cy, hdg, spd)
                mujoco.mj_step(m, d)

                collided = _check_collision(cx, cy, gvt_x, gvt_y)
                _record(gvt_x, gvt_y, collided)

                if collided and collision_time is None:
                    collision_time  = d.time
                    collision_speed = spd

                lp      = tracker.lookahead_point(cx, cy)
                lbl_pos = d.xpos[chassis_id].copy(); lbl_pos[2] += 3.0
                gvt_in_lane = gvt_x > 0
                ttc = (gvt_y - cy) / CLOSING_SPEED_MS if gvt_y > cy else 0.0
                draw_overlay(
                    viewer, PATH_EGO, lp, lbl_pos,
                    f"t={d.time:.2f}s  Ego={spd*3.6:.1f}km/h  "
                    f"sep={gvt_y-cy:.1f}m  TTC={ttc:.2f}s  "
                    f"GVT={'in lane' if gvt_in_lane else 'adj lane'}  "
                    f"{'COLLISION' if collided else ''}",
                )
                viewer.sync()
                sleep_t = m.opt.timestep - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            gvt_x, gvt_y = _update_mocap(d.time)
            pos = d.xpos[chassis_id]
            cx, cy = float(pos[0]), float(pos[1])
            hdg = heading_from_xmat(d.xmat, chassis_id)
            vx  = float(d.sensordata[vel_adr])
            vy  = float(d.sensordata[vel_adr + 1])
            spd = math.sqrt(vx**2 + vy**2)

            _step_controls(cx, cy, hdg, spd)
            mujoco.mj_step(m, d)

            collided = _check_collision(cx, cy, gvt_x, gvt_y)
            _record(gvt_x, gvt_y, collided)

            if collided and collision_time is None:
                collision_time  = d.time
                collision_speed = spd

    return {
        "time_s":           np.array(times),
        "x_m":              np.array(xs),
        "y_m":              np.array(ys),
        "speed_ms":         np.array(speeds),
        "accel_lon_ms2":    np.array(accel_lons),
        "accel_lat_ms2":    np.array(accel_lats),
        "steer_angle_rad":  np.array(steer_angles),
        "gvt_x_m":          np.array(gvt_xs),
        "gvt_y_m":          np.array(gvt_ys),
        "separation_m":     np.array(separations),
        "collision":        np.array(collision_flags),
        "collision_time_s": collision_time,
        "collision_speed_ms": collision_speed,
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict:
    times   = data["time_s"]
    speeds  = data["speed_ms"]
    gvt_xs  = data["gvt_x_m"]
    gvt_ys  = data["gvt_y_m"]
    ys      = data["y_m"]
    seps    = data["separation_m"]

    # Did GVT complete the lane change into Ego lane?
    gvt_in_ego_lane = np.any(gvt_xs > 0.0)
    gvt_final_x = float(gvt_xs[-1])

    # TTC at end of lane change (when GVT first fully enters Ego lane, x > 1.0)
    lc_done_mask = gvt_xs > 1.0
    if lc_done_mask.any():
        idx = int(np.argmax(lc_done_mask))
        ttc_at_lc_end = (float(gvt_ys[idx]) - float(ys[idx])) / CLOSING_SPEED_MS
    else:
        ttc_at_lc_end = float("nan")

    # Collision
    collision_detected = data["collision_time_s"] is not None
    collision_time     = data["collision_time_s"] or float("nan")
    collision_speed    = (data["collision_speed_ms"] * 3.6
                         if data["collision_speed_ms"] is not None else float("nan"))

    # Ego speed stats
    mean_ego_speed_kph = float(np.mean(speeds)) * 3.6
    peak_ego_speed_kph = float(np.max(speeds))  * 3.6

    return {
        "gvt_in_ego_lane":      gvt_in_ego_lane,
        "gvt_final_x_m":        gvt_final_x,
        "ttc_at_lc_end_s":      ttc_at_lc_end,
        "collision_detected":   collision_detected,
        "collision_time_s":     collision_time,
        "collision_speed_kph":  collision_speed,
        "mean_ego_speed_kph":   mean_ego_speed_kph,
        "peak_ego_speed_kph":   peak_ego_speed_kph,
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
              f"(expected ≈ {LANE_CHANGE_TTC_S:.1f} s)")
    if not math.isnan(metrics["collision_time_s"]):
        print(f"  [INFO] Collision time:           {metrics['collision_time_s']:.3f} s")
        print(f"  [INFO] Ego speed at collision:   {metrics['collision_speed_kph']:.1f} km/h")

    # Criterion 1: Ego holds 70 km/h (within 15 %)
    ok1 = 70.0 * 0.85 <= metrics["mean_ego_speed_kph"] <= 70.0 * 1.15
    ok  = ok and ok1
    print(f"\n  [{'PASS' if ok1 else 'FAIL'}] Ego maintains 70 km/h "
          f"(mean = {metrics['mean_ego_speed_kph']:.1f} km/h)")

    # Criterion 2: GVT completes lane change into Ego lane
    ok2 = metrics["gvt_in_ego_lane"]
    ok  = ok and ok2
    print(f"\n  [{'PASS' if ok2 else 'FAIL'}] GVT enters Ego lane "
          f"(final GVT x = {metrics['gvt_final_x_m']:.3f} m, "
          f"expected = {LANE_EGO_X:.2f} m)")

    # Criterion 3: TTC at end of lane change ≈ 1.5 s (±0.5 s tolerance)
    if math.isnan(metrics["ttc_at_lc_end_s"]):
        print("\n  [SKIP] TTC at lane change end: GVT never entered Ego lane")
    else:
        ok3 = abs(metrics["ttc_at_lc_end_s"] - LANE_CHANGE_TTC_S) <= 0.5
        ok  = ok and ok3
        print(f"\n  [{'PASS' if ok3 else 'FAIL'}] TTC at lane change end ≈ {LANE_CHANGE_TTC_S} s "
              f"(measured = {metrics['ttc_at_lc_end_s']:.2f} s)")

    # Criterion 4: Collision detected (expected — no AEB fitted)
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
        data["x_m"],            data["y_m"],
        data["speed_ms"],        data["speed_ms"] * 3.6,
        data["accel_lon_ms2"],   data["accel_lat_ms2"],
        data["steer_angle_rad"],
        data["gvt_x_m"],         data["gvt_y_m"],
        data["separation_m"],    data["collision"],
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
        description="NCAP AEB C2C CCFhol — head-on cut-in scenario (no AEB)")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open MuJoCo viewer during simulation")
    args = parser.parse_args()

    lc_gap_at_start = LC_START_TTC_S * CLOSING_SPEED_MS

    print("=" * 70)
    print("  NCAP AEB C2C CCFhol — Car-to-Car Front Head-On Cut-In Overlap")
    print("=" * 70)
    print(f"  Scene:              {SCENE_PATH}")
    print(f"  Model:              {MODEL_PATH}")
    print(f"  Ego speed:          {EGO_SPEED_KPH} km/h")
    print(f"  GVT speed:          {EGO_SPEED_KPH} km/h (southbound, head-on)")
    print(f"  GVT start:          x={LANE_GVT_X} m, y={GVT_START_Y} m")
    print(f"  LC distance:        {LANE_CHANGE_DIST_M} m")
    print(f"  LC duration:        {LC_DURATION_S:.3f} s")
    print(f"  LC start time:      t={LC_START_TIME_S:.3f} s  "
          f"(gap={lc_gap_at_start:.1f} m)")
    print(f"  Expected collision: t≈{TIME_TO_COLLISION_S:.2f} s")
    print(f"  Viewer:             {'on' if args.viewer else 'off (pass --viewer)'}")

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
