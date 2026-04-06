"""
Scenario 1 — Straight-Through Junction Traversal
=================================================
The ego vehicle starts in the south arm (northbound lane, x=+1.75 m, y=-35 m),
accelerates to a cruising speed, and drives straight through the 4-way junction,
exiting via the north arm.

Control phases (position-based):
  Phase 1  y < -15 m  : throttle=0.30, brake=0  — accelerate to ~50 km/h
  Phase 2  y >= -15 m : throttle=0.12, brake=0  — maintain cruising speed
  Throughout          : steering=0               — straight ahead

Pass criteria:
  1. Vehicle exits the north arm  (chassis y > +20 m)
  2. Vehicle stays in northbound lane  (|chassis x - 1.75| < 1.0 m at all times)
  3. Minimum speed through the junction box (|y| < 7 m) >= 20 km/h

Usage
-----
Headless:
  mjpython scenarios/straight_through/index.py

With MuJoCo viewer:
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

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent.parent
SCENE_PATH  = ROOT_DIR / "scenes" / "junction.xml"
MODEL_PATH  = ROOT_DIR / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "straight_through.csv"

# ── Scenario parameters ────────────────────────────────────────────────────────
THROTTLE_ACCEL   = 0.30   # throttle during acceleration phase
THROTTLE_CRUISE  = 0.12   # throttle to maintain speed
ACCEL_PHASE_Y    = -15.0  # m — switch accel→cruise once chassis y > this
TEST_DURATION_S  = 25.0   # s — simulation time budget

# ── Pass criteria ──────────────────────────────────────────────────────────────
EXIT_Y_M         = 20.0   # chassis must reach this y to be considered "exited"
LANE_CENTRE_X    = 1.75   # m — northbound lane centre
LANE_TOL_M       = 1.0    # m — maximum lateral deviation from lane centre
MIN_JUNCTION_KPH = 20.0   # km/h — minimum speed required while inside junction box
JUNCTION_BOX     = 7.0    # m — |y| < this defines the junction box


# ── Simulation ──────────────────────────────────────────────────────────────────

def build_model() -> mujoco.MjModel:
    scene_spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    robot_spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    scene_spec.attach(robot_spec, frame="world", prefix="robot-")
    return scene_spec.compile()


def run_simulation(m: mujoco.MjModel, show_viewer: bool = False) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    # ── IDs ───────────────────────────────────────────────────────────────────
    steering_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot-steering")
    throttle_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot-throttle")
    brake_id     = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot-brake")
    vel_id       = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "robot-velocimeter")
    acc_id       = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "robot-accelerometer")
    chassis_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     "robot-chassis")
    fl_jnt_id    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,    "robot-wheel_fl_steering")

    vel_adr   = int(m.sensor_adr[vel_id])
    acc_adr   = int(m.sensor_adr[acc_id])
    fl_qpos   = int(m.jnt_qposadr[fl_jnt_id])

    # ── Telemetry buffers ─────────────────────────────────────────────────────
    times, xs, ys = [], [], []
    speeds, accel_lons, accel_lats = [], [], []
    steer_angles, phases = [], []

    def _apply_controls():
        chassis_y = float(d.xpos[chassis_id][1])
        phase = "accel" if chassis_y < ACCEL_PHASE_Y else "cruise"
        throttle = THROTTLE_ACCEL if phase == "accel" else THROTTLE_CRUISE
        d.ctrl[steering_id] = 0.0
        d.ctrl[throttle_id] = throttle
        d.ctrl[brake_id]    = 0.0
        return phase

    def _record(phase: str):
        vx    = float(d.sensordata[vel_adr])
        vy    = float(d.sensordata[vel_adr + 1])
        lon_a = float(d.sensordata[acc_adr])
        lat_a = float(d.sensordata[acc_adr + 1])
        steer = float(d.qpos[fl_qpos])
        pos   = d.xpos[chassis_id]

        times.append(d.time)
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        speeds.append(math.sqrt(vx**2 + vy**2))
        accel_lons.append(lon_a)
        accel_lats.append(lat_a)
        steer_angles.append(steer)
        phases.append(phase)

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                step_start = time.time()
                phase = _apply_controls()
                mujoco.mj_step(m, d)
                _record(phase)

                chassis_y = float(d.xpos[chassis_id][1])
                if chassis_y > EXIT_Y_M + 10:
                    break

                speed_kph = speeds[-1] * 3.6
                viewer.user_scn.ngeom = 0
                geom = viewer.user_scn.geoms[0]
                label_pos = d.xpos[chassis_id].copy()
                label_pos[2] += 3.0
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LABEL,
                    np.zeros(3),
                    label_pos,
                    np.eye(3).flatten(),
                    np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
                )
                geom.label = (
                    f"t={d.time:.1f}s  "
                    f"v={speed_kph:.1f} km/h  "
                    f"x={xs[-1]:.2f} m  y={ys[-1]:.2f} m  "
                    f"[{phase}]"
                )
                viewer.user_scn.ngeom = 1
                viewer.sync()
                elapsed = time.time() - step_start
                sleep_t = m.opt.timestep - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            phase = _apply_controls()
            mujoco.mj_step(m, d)
            _record(phase)

            chassis_y = float(d.xpos[chassis_id][1])
            if chassis_y > EXIT_Y_M + 10:
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


# ── Metrics & checks ────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict:
    xs      = data["x_m"]
    ys      = data["y_m"]
    speeds  = data["speed_ms"]

    exited = bool(np.any(ys > EXIT_Y_M))

    lat_dev     = np.abs(xs - LANE_CENTRE_X)
    max_lat_dev = float(np.max(lat_dev))

    in_box = np.abs(ys) < JUNCTION_BOX
    if in_box.any():
        min_junction_speed_kph = float(np.min(speeds[in_box])) * 3.6
    else:
        min_junction_speed_kph = float("nan")

    peak_speed_kph = float(np.max(speeds)) * 3.6

    return {
        "exited":                  exited,
        "max_lateral_deviation_m": max_lat_dev,
        "min_junction_speed_kph":  min_junction_speed_kph,
        "peak_speed_kph":          peak_speed_kph,
        "final_y_m":               float(ys[-1]),
    }


# ── Report ──────────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    all_pass = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")
    print(f"  [INFO] Final chassis y:          {metrics['final_y_m']:.2f} m")
    print(f"  [INFO] Peak speed:               {metrics['peak_speed_kph']:.1f} km/h")
    print(f"  [INFO] Max lateral deviation:    {metrics['max_lateral_deviation_m']:.3f} m")
    print(f"  [INFO] Min speed in junction:    {metrics['min_junction_speed_kph']:.1f} km/h")

    # Check 1: vehicle exited north arm
    ok1 = metrics["exited"]
    if not ok1:
        all_pass = False
    print(f"\n  [{'PASS' if ok1 else 'FAIL'}] Vehicle exits north arm (y > {EXIT_Y_M} m)")
    print(f"         final y = {metrics['final_y_m']:.2f} m")

    # Check 2: stays in lane
    ok2 = metrics["max_lateral_deviation_m"] <= LANE_TOL_M
    if not ok2:
        all_pass = False
    print(f"\n  [{'PASS' if ok2 else 'FAIL'}] Stays in northbound lane "
          f"(|x - {LANE_CENTRE_X}| ≤ {LANE_TOL_M} m)")
    print(f"         max deviation = {metrics['max_lateral_deviation_m']:.3f} m")

    # Check 3: minimum speed through junction
    if math.isnan(metrics["min_junction_speed_kph"]):
        print(f"\n  [SKIP] Junction speed check: vehicle never entered junction box")
    else:
        ok3 = metrics["min_junction_speed_kph"] >= MIN_JUNCTION_KPH
        if not ok3:
            all_pass = False
        print(f"\n  [{'PASS' if ok3 else 'FAIL'}] Minimum junction speed "
              f"≥ {MIN_JUNCTION_KPH} km/h")
        print(f"         min speed in box = {metrics['min_junction_speed_kph']:.1f} km/h")

    return all_pass


# ── CSV ─────────────────────────────────────────────────────────────────────────

def save_csv(data: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([
        data["time_s"],
        data["x_m"],
        data["y_m"],
        data["speed_ms"],
        data["speed_ms"] * 3.6,
        data["accel_lon_ms2"],
        data["accel_lat_ms2"],
        data["steer_angle_rad"],
    ])
    np.savetxt(
        CSV_PATH, rows, delimiter=",",
        header="time_s,x_m,y_m,speed_ms,speed_kph,accel_lon_ms2,accel_lat_ms2,steer_angle_rad",
        comments="",
    )
    print(f"\n  CSV saved → {CSV_PATH}")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scenario 1: Straight-through junction traversal")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open the MuJoCo viewer during simulation")
    args = parser.parse_args()

    print("=" * 70)
    print("  Scenario 1 — Straight-Through Junction Traversal")
    print("=" * 70)
    print(f"  Scene:   {SCENE_PATH}")
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Spawn:   x={LANE_CENTRE_X} m, y=-35 m, heading=north")
    print(f"  Viewer:  {'on' if args.viewer else 'off (pass --viewer to enable)'}")

    for path, label in [(SCENE_PATH, "scene"), (MODEL_PATH, "model")]:
        if not path.exists():
            print(f"\n  ERROR: {label} file not found at {path}")
            sys.exit(1)

    m = build_model()

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
