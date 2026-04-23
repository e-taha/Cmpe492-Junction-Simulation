"""
Scenario 2 — Right Turn at Junction (South → East)
====================================================
The ego vehicle starts in the south arm (northbound lane, x=+1.75 m, y=-35 m),
approaches the junction, decelerates, and turns right into the eastbound lane
(y=-1.75 m), then exits via the east arm.

Reference path
--------------
  Segment 1 — straight approach:
    (1.75, -35) → (1.75, -7)   along northbound lane

  Segment 2 — circular arc (right turn, clockwise):
    centre (7.0, -7.0), radius 5.25 m
    θ: π → π/2  (clockwise)
    entry  (1.75, -7.0)  heading north
    exit   (7.0,  -1.75) heading east

  Segment 3 — straight exit:
    (7.0, -1.75) → (45.0, -1.75)  along eastbound lane

  Arc geometry derivation:
    Northbound lane x = +1.75 m → right = +x direction
    Arc centre = (1.75 + R, -7) = (7.0, -7)  with R = 5.25 m
    After 90° CW turn: exit point = centre + (0, +R) = (7.0, -1.75) ✓

  Tracked by Pure Pursuit (lookahead = 3 m).

Throttle phases (position-based; steering is always Pure Pursuit):
  approach  y < -10 m        : throttle=0.18
  brake     -10 ≤ y < -7 m  : brake=0.45
  maneuver  x <  12 m       : throttle=0.10
  exit      x ≥  12 m       : throttle=0.22

Pass criteria:
  1. Vehicle enters east arm        (chassis x > +15 m)
  2. Exits in eastbound lane        (-7 m ≤ chassis y ≤ 0 m while x > +7 m)
  3. Does not mount the inner kerb  (chassis y ≥ -7 m throughout)

Usage
-----
  mjpython scenarios/right_turn/index.py
  mjpython scenarios/right_turn/index.py --viewer
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
                          make_path, heading_from_xmat, draw_overlay)

# ── File paths ─────────────────────────────────────────────────────────────────
SCENE_PATH  = ROOT_DIR / "scenes" / "junction.xml"
MODEL_PATH  = ROOT_DIR / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "right_turn.csv"

# ── Scenario parameters ────────────────────────────────────────────────────────
WHEELBASE_M    = 3.128
LOOKAHEAD_M    = 3.0     # m — short lookahead for tight turn tracking
TEST_DURATION_S = 30.0   # s

# ── Pass criteria ──────────────────────────────────────────────────────────────
EAST_ARM_X  = 15.0   # m
LANE_Y_MIN  = -7.0   # m — southern edge of eastbound lane
LANE_Y_MAX  =  0.0   # m — centre line (upper edge of eastbound lane)
JUNCTION_BOX = 7.0   # m

# ── Reference path ─────────────────────────────────────────────────────────────
#
#              N
#              ↑
#              │  x=1.75 (northbound)
#   (7,-1.75) ─────────────────────────────→ E   y=-1.75 (eastbound)
#          ╮  arc centre (7,-7), R=5.25 m
#          │
#          │  (1.75,-7) to (1.75,-35)
#          ↓
#        spawn
#
_ARC_CX, _ARC_CY, _ARC_R = 7.0, -7.0, 5.25

PATH = make_path(
    straight_segment(1.75, -35.0,  1.75,  -7.0),          # approach
    arc_segment(_ARC_CX, _ARC_CY, _ARC_R,                  # right turn
                math.pi, math.pi / 2),                     # θ: π → π/2 (CW)
    straight_segment(7.0,  -1.75, 45.0,  -1.75),           # exit east
)


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

        # Throttle/brake phase (position-based)
        if cy < -10.0:
            phase = "approach"
            d.ctrl[throttle_id] = 0.18
            d.ctrl[brake_id]    = 0.0
        elif cy < -7.0:
            phase = "brake"
            d.ctrl[throttle_id] = 0.0
            d.ctrl[brake_id]    = 0.45
        elif cx < 12.0:
            phase = "maneuver"
            d.ctrl[throttle_id] = 0.10
            d.ctrl[brake_id]    = 0.0
        else:
            phase = "exit"
            d.ctrl[throttle_id] = 0.22
            d.ctrl[brake_id]    = 0.0

        # Steering always from Pure Pursuit
        d.ctrl[steering_id] = tracker.compute_ctrl(cx, cy, hdg)
        return phase

    def _record(phase: str):
        vx  = float(d.sensordata[vel_adr])
        vy  = float(d.sensordata[vel_adr + 1])
        pos = d.xpos[chassis_id]
        times.append(d.time)
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        speeds.append(math.sqrt(vx**2 + vy**2))
        accel_lons.append(float(d.sensordata[acc_adr]))
        accel_lats.append(float(d.sensordata[acc_adr + 1]))
        steer_angles.append(float(d.qpos[fl_qpos]))
        phases.append(phase)

    def _done() -> bool:
        return float(d.xpos[chassis_id][0]) > EAST_ARM_X + 15

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
            phase = _step_controls()
            mujoco.mj_step(m, d)
            _record(phase)
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
    phases = np.array(data["phase"])

    entered_east = bool(np.any(xs > EAST_ARM_X))

    in_east = xs > JUNCTION_BOX
    y_east_min = float(np.min(ys[in_east])) if in_east.any() else float("nan")
    y_east_max = float(np.max(ys[in_east])) if in_east.any() else float("nan")

    # Kerb check: minimum y during the maneuver phase
    maneuvering = phases == "maneuver"
    min_y_maneuver = float(np.min(ys[maneuvering])) if maneuvering.any() else float("nan")
    spd_maneuver   = float(np.mean(speeds[maneuvering])) * 3.6 if maneuvering.any() else float("nan")

    return {
        "entered_east":      entered_east,
        "y_east_min":        y_east_min,
        "y_east_max":        y_east_max,
        "min_y_maneuver":    min_y_maneuver,
        "turn_speed_kph":    spd_maneuver,
        "peak_speed_kph":    float(np.max(speeds)) * 3.6,
        "final_x_m":         float(xs[-1]),
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    ok = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")
    print(f"  [INFO] Final chassis x:         {metrics['final_x_m']:.2f} m")
    print(f"  [INFO] Peak speed:              {metrics['peak_speed_kph']:.1f} km/h")
    print(f"  [INFO] Mean speed in maneuver:  {metrics['turn_speed_kph']:.1f} km/h")
    print(f"  [INFO] Min y during maneuver:   {metrics['min_y_maneuver']:.2f} m")
    if not math.isnan(metrics["y_east_min"]):
        print(f"  [INFO] y range in east arm:     "
              f"[{metrics['y_east_min']:.2f}, {metrics['y_east_max']:.2f}] m")

    ok1 = metrics["entered_east"]
    ok  = ok and ok1
    print(f"\n  [{'PASS' if ok1 else 'FAIL'}] Vehicle enters east arm (x > {EAST_ARM_X} m)")
    print(f"         final x = {metrics['final_x_m']:.2f} m")

    if math.isnan(metrics["y_east_min"]):
        print(f"\n  [SKIP] Eastbound lane check: vehicle never crossed junction box")
    else:
        ok2 = metrics["y_east_min"] >= LANE_Y_MIN and metrics["y_east_max"] <= LANE_Y_MAX
        ok  = ok and ok2
        print(f"\n  [{'PASS' if ok2 else 'FAIL'}] Exits in eastbound lane "
              f"({LANE_Y_MIN} ≤ y ≤ {LANE_Y_MAX} m)")
        print(f"         y range in east arm = "
              f"[{metrics['y_east_min']:.2f}, {metrics['y_east_max']:.2f}] m")

    if math.isnan(metrics["min_y_maneuver"]):
        print(f"\n  [SKIP] Kerb check: no maneuver phase recorded")
    else:
        ok3 = metrics["min_y_maneuver"] >= LANE_Y_MIN
        ok  = ok and ok3
        print(f"\n  [{'PASS' if ok3 else 'FAIL'}] Stays within road (y ≥ {LANE_Y_MIN} m)")
        print(f"         min y during maneuver = {metrics['min_y_maneuver']:.2f} m")

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
        description="Scenario 2: Right-turn junction traversal (Pure Pursuit)")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open the MuJoCo viewer during simulation")
    args = parser.parse_args()

    print("=" * 70)
    print("  Scenario 2 — Right Turn at Junction (South → East)")
    print("=" * 70)
    print(f"  Scene:       {SCENE_PATH}")
    print(f"  Model:       {MODEL_PATH}")
    print(f"  Spawn:       x=1.75 m, y=-35 m, heading=north")
    print(f"  Arc:         centre ({_ARC_CX}, {_ARC_CY}), R={_ARC_R} m")
    print(f"  Lookahead:   {LOOKAHEAD_M} m")
    print(f"  Path pts:    {len(PATH)}")
    print(f"  Viewer:      {'on' if args.viewer else 'off (pass --viewer to enable)'}")

    for p, label in [(SCENE_PATH, "scene"), (MODEL_PATH, "model")]:
        if not p.exists():
            print(f"\n  ERROR: {label} file not found: {p}")
            sys.exit(1)

    m = build_model()
    print("\n  Running simulation...")
    data    = run_simulation(m, show_viewer=args.viewer)
    metrics = extract_metrics(data)
    ok      = print_report(metrics)
    save_csv(data)

    print("\n" + "=" * 70)
    print(f"  {'PASS' if ok else 'FAIL'} — right-turn scenario")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
