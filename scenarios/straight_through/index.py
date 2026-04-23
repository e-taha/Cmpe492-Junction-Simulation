"""
Scenario 1 — Straight-Through Junction Traversal
=================================================
The ego vehicle starts in the south arm (northbound lane, x=+1.75 m, y=-35 m),
accelerates to cruising speed, and drives straight through the 4-way junction,
exiting via the north arm.

Reference path
--------------
  Straight line along x = +1.75 m (northbound lane centre) from y=-35 to y=+42.
  Tracked by the Pure Pursuit controller (lookahead = 6 m).

Throttle phases (position-based; steering is always Pure Pursuit):
  accel   y < -15 m   : throttle=0.30
  cruise  y ≥ -15 m   : throttle=0.12

Pass criteria:
  1. Vehicle exits the north arm  (chassis y > +20 m)
  2. Stays in northbound lane     (|chassis x - 1.75| ≤ 1.0 m at all times)
  3. Minimum speed through junction box (|y| < 7 m) ≥ 20 km/h

Usage
-----
  mjpython scenarios/straight_through/index.py
  mjpython scenarios/straight_through/index.py --viewer
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ── Module path so we can import src/pure_pursuit.py ──────────────────────────
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from pure_pursuit import (PurePursuit, straight_segment,
                          heading_from_xmat, draw_overlay)

# ── File paths ─────────────────────────────────────────────────────────────────
SCENE_PATH  = ROOT_DIR / "scenes" / "junction.xml"
MODEL_PATH  = ROOT_DIR / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "straight_through.csv"

# ── Scenario parameters ────────────────────────────────────────────────────────
WHEELBASE_M     = 3.128
LOOKAHEAD_M     = 6.0    # m — lookahead distance for Pure Pursuit
THROTTLE_ACCEL  = 0.30
THROTTLE_CRUISE = 0.12
ACCEL_PHASE_Y   = -15.0  # m — switch accel→cruise once chassis y > this
TEST_DURATION_S = 25.0   # s

# ── Pass criteria ──────────────────────────────────────────────────────────────
EXIT_Y_M         = 20.0
LANE_CENTRE_X    = 1.75
LANE_TOL_M       = 1.0
MIN_JUNCTION_KPH = 20.0
JUNCTION_BOX_M   = 7.0

# ── Reference path ─────────────────────────────────────────────────────────────
#
#   N
#   ↑                    x = 1.75 (northbound lane)
#   │   ┌──────────────────────────── y = +42
#   │   │
#   │   │   (straight, heading north)
#   │   │
#   │   └──────────────────────────── y = -35  (spawn)
#
PATH = straight_segment(1.75, -35.0, 1.75, 42.0)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> mujoco.MjModel:
    scene_spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    robot_spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    scene_spec.attach(robot_spec, frame="world", prefix="robot-")
    return scene_spec.compile()


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel, show_viewer: bool = False) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    steering_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot-steering")
    throttle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot-throttle")
    brake_id    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot-brake")
    vel_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "robot-velocimeter")
    acc_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "robot-accelerometer")
    chassis_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     "robot-chassis")
    fl_jnt_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,    "robot-wheel_fl_steering")

    vel_adr = int(m.sensor_adr[vel_id])
    acc_adr = int(m.sensor_adr[acc_id])
    fl_qpos = int(m.jnt_qposadr[fl_jnt_id])

    tracker = PurePursuit(PATH, WHEELBASE_M, LOOKAHEAD_M)

    times, xs, ys           = [], [], []
    speeds, accel_lons, accel_lats = [], [], []
    steer_angles, phases    = [], []

    def _step_controls() -> str:
        cx  = float(d.xpos[chassis_id][0])
        cy  = float(d.xpos[chassis_id][1])
        hdg = heading_from_xmat(d.xmat, chassis_id)

        phase    = "accel" if cy < ACCEL_PHASE_Y else "cruise"
        throttle = THROTTLE_ACCEL if phase == "accel" else THROTTLE_CRUISE

        d.ctrl[steering_id] = tracker.compute_ctrl(cx, cy, hdg)
        d.ctrl[throttle_id] = throttle
        d.ctrl[brake_id]    = 0.0
        return phase

    def _record(phase: str):
        vx    = float(d.sensordata[vel_adr])
        vy    = float(d.sensordata[vel_adr + 1])
        pos   = d.xpos[chassis_id]
        times.append(d.time)
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        speeds.append(math.sqrt(vx**2 + vy**2))
        accel_lons.append(float(d.sensordata[acc_adr]))
        accel_lats.append(float(d.sensordata[acc_adr + 1]))
        steer_angles.append(float(d.qpos[fl_qpos]))
        phases.append(phase)

    def _done() -> bool:
        return float(d.xpos[chassis_id][1]) > EXIT_Y_M + 10

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                t0    = time.time()
                phase = _step_controls()
                mujoco.mj_step(m, d)
                _record(phase)
                if _done():
                    break

                cx, cy = xs[-1], ys[-1]
                hdg    = heading_from_xmat(d.xmat, chassis_id)
                lp     = tracker.lookahead_point(cx, cy)
                lbl    = d.xpos[chassis_id].copy(); lbl[2] += 3.0
                draw_overlay(
                    viewer, PATH, lp, lbl,
                    f"t={d.time:.1f}s  v={speeds[-1]*3.6:.1f} km/h  "
                    f"x={cx:.2f} m  y={cy:.2f} m  [{phase}]",
                )
                viewer.sync()
                sleep_t = m.opt.timestep - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            _step_controls()
            mujoco.mj_step(m, d)
            _record("accel" if float(d.xpos[chassis_id][1]) < ACCEL_PHASE_Y else "cruise")
            if _done():
                break

    return {
        "time_s":          np.array(times),
        "x_m":             np.array(xs),
        "y_m":             np.array(ys),
        "speed_ms":        np.array(speeds),
        "accel_lon_ms2":   np.array(accel_lons),
        "accel_lat_ms2":   np.array(accel_lats),
        "steer_angle_rad": np.array(steer_angles),
        "phase":           phases,
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict:
    xs, ys, speeds = data["x_m"], data["y_m"], data["speed_ms"]
    exited     = bool(np.any(ys > EXIT_Y_M))
    max_lat    = float(np.max(np.abs(xs - LANE_CENTRE_X)))
    in_box     = np.abs(ys) < JUNCTION_BOX_M
    min_jct_kph = float(np.min(speeds[in_box])) * 3.6 if in_box.any() else float("nan")
    return {
        "exited":                  exited,
        "max_lateral_deviation_m": max_lat,
        "min_junction_speed_kph":  min_jct_kph,
        "peak_speed_kph":          float(np.max(speeds)) * 3.6,
        "final_y_m":               float(ys[-1]),
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    ok = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")
    print(f"  [INFO] Final chassis y:        {metrics['final_y_m']:.2f} m")
    print(f"  [INFO] Peak speed:             {metrics['peak_speed_kph']:.1f} km/h")
    print(f"  [INFO] Max lateral deviation:  {metrics['max_lateral_deviation_m']:.3f} m")
    print(f"  [INFO] Min speed in junction:  {metrics['min_junction_speed_kph']:.1f} km/h")

    ok1 = metrics["exited"]
    ok  = ok and ok1
    print(f"\n  [{'PASS' if ok1 else 'FAIL'}] Vehicle exits north arm (y > {EXIT_Y_M} m)")
    print(f"         final y = {metrics['final_y_m']:.2f} m")

    ok2 = metrics["max_lateral_deviation_m"] <= LANE_TOL_M
    ok  = ok and ok2
    print(f"\n  [{'PASS' if ok2 else 'FAIL'}] Stays in northbound lane "
          f"(|x - {LANE_CENTRE_X}| ≤ {LANE_TOL_M} m)")
    print(f"         max deviation = {metrics['max_lateral_deviation_m']:.3f} m")

    if math.isnan(metrics["min_junction_speed_kph"]):
        print(f"\n  [SKIP] Junction speed: vehicle never entered junction box")
    else:
        ok3 = metrics["min_junction_speed_kph"] >= MIN_JUNCTION_KPH
        ok  = ok and ok3
        print(f"\n  [{'PASS' if ok3 else 'FAIL'}] Min junction speed ≥ {MIN_JUNCTION_KPH} km/h")
        print(f"         min speed in box = {metrics['min_junction_speed_kph']:.1f} km/h")

    return ok


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(data: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([
        data["time_s"], data["x_m"], data["y_m"],
        data["speed_ms"], data["speed_ms"] * 3.6,
        data["accel_lon_ms2"], data["accel_lat_ms2"], data["steer_angle_rad"],
    ])
    np.savetxt(CSV_PATH, rows, delimiter=",",
               header="time_s,x_m,y_m,speed_ms,speed_kph,"
                      "accel_lon_ms2,accel_lat_ms2,steer_angle_rad",
               comments="")
    print(f"\n  CSV saved → {CSV_PATH}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scenario 1: Straight-through junction traversal (Pure Pursuit)")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open the MuJoCo viewer during simulation")
    args = parser.parse_args()

    print("=" * 70)
    print("  Scenario 1 — Straight-Through Junction Traversal")
    print("=" * 70)
    print(f"  Scene:      {SCENE_PATH}")
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Spawn:      x={LANE_CENTRE_X} m, y=-35 m, heading=north")
    print(f"  Lookahead:  {LOOKAHEAD_M} m")
    print(f"  Path pts:   {len(PATH)}")
    print(f"  Viewer:     {'on' if args.viewer else 'off (pass --viewer to enable)'}")

    for p, label in [(SCENE_PATH, "scene"), (MODEL_PATH, "model")]:
        if not p.exists():
            print(f"\n  ERROR: {label} file not found: {p}")
            sys.exit(1)

    m    = build_model()
    print("\n  Running simulation...")
    data    = run_simulation(m, show_viewer=args.viewer)
    metrics = extract_metrics(data)
    ok      = print_report(metrics)
    save_csv(data)

    print("\n" + "=" * 70)
    print(f"  {'PASS' if ok else 'FAIL'} — straight-through scenario")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
