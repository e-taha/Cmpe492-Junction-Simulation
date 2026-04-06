"""
Scenario 2 — Right Turn at Junction (South → East)
====================================================
The ego vehicle starts in the south arm (northbound lane, x=+1.75 m, y=-35 m),
approaches the junction, decelerates, and turns right into the eastbound lane
(y=-1.75 m), then exits via the east arm.

Junction geometry (right-hand traffic):
  - Northbound lane centre : x = +1.75 m
  - Eastbound lane centre  : y = -1.75 m
  - Junction box           : |x| < 7 m and |y| < 7 m
  - Ideal turn arc centre  : (x=+7, y=-7)  — inner corner
  - Ideal turn radius      : 5.25 m
  - Required wheel angle   : delta = atan(L / R) = atan(3.128 / 5.25) ≈ 0.538 rad
  - Steering ctrl (right)  : +2 × 0.538 ≈ +1.076

Control phases (position + heading based):
  Phase 1  y < -20 m          : throttle=0.20, steer=0    — accelerate to ~30 km/h
  Phase 2  -20 ≤ y < -9 m     : throttle=0.08, steer=0    — hold moderate speed
  Phase 3  -9 ≤ y < -7 m      : throttle=0,   brake=0.55  — brake before turn
  Phase 4  turning (hdg N→E)   : throttle=0.10, steer=+1.076 — execute right turn
  Phase 5  heading ≈ east      : throttle=0.22, steer=0    — straighten and exit east

Pass criteria:
  1. Vehicle enters east arm  (chassis x > +15 m)
  2. Exits in eastbound lane  (chassis y between -7 m and  0 m once x > +7 m)
  3. Does not mount the kerb  (chassis y > -7 m throughout the turn)

Usage
-----
Headless:
  mjpython scenarios/right_turn/index.py

With MuJoCo viewer:
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

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent.parent
SCENE_PATH  = ROOT_DIR / "scenes" / "junction.xml"
MODEL_PATH  = ROOT_DIR / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "right_turn.csv"

# ── Vehicle / junction constants ───────────────────────────────────────────────
WHEELBASE_M      = 3.128          # m
JUNCTION_BOX     = 7.0            # m — |coord| < this = inside junction
NORTHBOUND_X     = 1.75           # m — northbound lane centre
EASTBOUND_Y      = -1.75          # m — eastbound lane centre
TURN_RADIUS_M    = 5.25           # m — arc radius for ideal right turn
TURN_DELTA_RAD   = math.atan(WHEELBASE_M / TURN_RADIUS_M)  # ≈ 0.538 rad
STEER_RIGHT_CTRL = 2.0 * TURN_DELTA_RAD                    # ≈ +1.076

# ── Phase thresholds ──────────────────────────────────────────────────────────
Y_HOLD_SPEED   = -20.0   # m — switch to hold-speed phase below this y
Y_BRAKE_START  = -9.0    # m — begin braking
Y_TURN_START   = -7.0    # m — begin right-turn steering
HDG_EAST_RAD   = 0.15    # rad — heading threshold to declare "facing east"
TEST_DURATION_S = 30.0   # s

# ── Pass criteria ──────────────────────────────────────────────────────────────
EAST_ARM_X     = 15.0    # m — x must exceed this to count as "entered east arm"
LANE_Y_MIN     = -7.0    # m — lower bound of eastbound lane (road edge)
LANE_Y_MAX     =  0.0    # m — upper bound of eastbound lane (centre line)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _heading_rad(xmat: np.ndarray, body_id: int) -> float:
    """Yaw heading of body in world frame (atan2 of forward direction).

    d.xmat has shape (nbody, 9): each row is the flattened 3×3 rotation matrix
    for that body. Column 0 = world-x of local +x, column 1 = world-y of local +x.
    """
    # xmat row layout (row-major 3×3): [R00 R01 R02 R10 R11 R12 R20 R21 R22]
    # Forward direction = local +x → world vector = first column of R:
    #   world-x = R[0,0] = index 0
    #   world-y = R[1,0] = index 3
    fx = float(xmat[body_id, 0])
    fy = float(xmat[body_id, 3])
    return math.atan2(fy, fx)


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

    # ── IDs ───────────────────────────────────────────────────────────────────
    steering_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot-steering")
    throttle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot-throttle")
    brake_id    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot-brake")
    vel_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "robot-velocimeter")
    acc_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "robot-accelerometer")
    chassis_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     "robot-chassis")
    fl_jnt_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,    "robot-wheel_fl_steering")

    vel_adr  = int(m.sensor_adr[vel_id])
    acc_adr  = int(m.sensor_adr[acc_id])
    fl_qpos  = int(m.jnt_qposadr[fl_jnt_id])

    # ── Telemetry buffers ─────────────────────────────────────────────────────
    times, xs, ys           = [], [], []
    speeds                  = []
    accel_lons, accel_lats  = [], []
    steer_angles, phases    = [], []

    def _phase_and_controls() -> str:
        cx  = float(d.xpos[chassis_id][0])
        cy  = float(d.xpos[chassis_id][1])
        hdg = _heading_rad(d.xmat, chassis_id)

        if cy < Y_HOLD_SPEED:
            phase = "accel"
            d.ctrl[steering_id] = 0.0
            d.ctrl[throttle_id] = 0.20
            d.ctrl[brake_id]    = 0.0
        elif cy < Y_BRAKE_START:
            phase = "hold"
            d.ctrl[steering_id] = 0.0
            d.ctrl[throttle_id] = 0.08
            d.ctrl[brake_id]    = 0.0
        elif cy < Y_TURN_START:
            phase = "brake"
            d.ctrl[steering_id] = 0.0
            d.ctrl[throttle_id] = 0.0
            d.ctrl[brake_id]    = 0.55
        elif hdg > HDG_EAST_RAD:
            # Still turning — heading not yet east
            phase = "turn"
            d.ctrl[steering_id] = STEER_RIGHT_CTRL
            d.ctrl[throttle_id] = 0.10
            d.ctrl[brake_id]    = 0.0
        else:
            # Heading is approximately east — straighten and drive out
            phase = "exit"
            d.ctrl[steering_id] = 0.0
            d.ctrl[throttle_id] = 0.22
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

    def _done() -> bool:
        return float(d.xpos[chassis_id][0]) > EAST_ARM_X + 15

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                step_start = time.time()
                phase = _phase_and_controls()
                mujoco.mj_step(m, d)
                _record(phase)
                if _done():
                    break

                speed_kph = speeds[-1] * 3.6
                hdg_deg   = math.degrees(_heading_rad(d.xmat, chassis_id))
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
                    f"hdg={hdg_deg:.1f}°  [{phase}]"
                )
                viewer.user_scn.ngeom = 1
                viewer.sync()
                elapsed = time.time() - step_start
                sleep_t = m.opt.timestep - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            phase = _phase_and_controls()
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


# ── Metrics & checks ────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict:
    xs     = data["x_m"]
    ys     = data["y_m"]
    speeds = data["speed_ms"]

    entered_east = bool(np.any(xs > EAST_ARM_X))

    # y position while in east arm (x > junction box)
    in_east = xs > JUNCTION_BOX
    if in_east.any():
        y_in_east_min = float(np.min(ys[in_east]))
        y_in_east_max = float(np.max(ys[in_east]))
    else:
        y_in_east_min = float("nan")
        y_in_east_max = float("nan")

    # Minimum y (southernmost) during the turn — must stay above -7 (road edge)
    turning = np.array(data["phase"]) == "turn"
    min_y_during_turn = float(np.min(ys[turning])) if turning.any() else float("nan")

    peak_speed_kph   = float(np.max(speeds)) * 3.6
    turn_speed_kph   = float(np.mean(speeds[turning])) * 3.6 if turning.any() else float("nan")

    return {
        "entered_east":      entered_east,
        "y_in_east_min":     y_in_east_min,
        "y_in_east_max":     y_in_east_max,
        "min_y_during_turn": min_y_during_turn,
        "peak_speed_kph":    peak_speed_kph,
        "turn_speed_kph":    turn_speed_kph,
        "final_x_m":         float(xs[-1]),
    }


# ── Report ──────────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    all_pass = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")
    print(f"  [INFO] Final chassis x:          {metrics['final_x_m']:.2f} m")
    print(f"  [INFO] Peak speed:               {metrics['peak_speed_kph']:.1f} km/h")
    print(f"  [INFO] Mean speed during turn:   {metrics['turn_speed_kph']:.1f} km/h")
    print(f"  [INFO] Min y during turn:        {metrics['min_y_during_turn']:.2f} m")
    if not math.isnan(metrics["y_in_east_min"]):
        print(f"  [INFO] y range in east arm:      "
              f"[{metrics['y_in_east_min']:.2f}, {metrics['y_in_east_max']:.2f}] m")

    # Check 1: vehicle entered east arm
    ok1 = metrics["entered_east"]
    if not ok1:
        all_pass = False
    print(f"\n  [{'PASS' if ok1 else 'FAIL'}] Vehicle enters east arm (x > {EAST_ARM_X} m)")
    print(f"         final x = {metrics['final_x_m']:.2f} m")

    # Check 2: y in correct eastbound lane while in east arm
    if math.isnan(metrics["y_in_east_min"]):
        print(f"\n  [SKIP] Eastbound lane check: vehicle never entered east arm")
    else:
        ok2 = (metrics["y_in_east_min"] >= LANE_Y_MIN and
               metrics["y_in_east_max"] <= LANE_Y_MAX)
        if not ok2:
            all_pass = False
        print(f"\n  [{'PASS' if ok2 else 'FAIL'}] Exits in eastbound lane "
              f"({LANE_Y_MIN} m ≤ y ≤ {LANE_Y_MAX} m)")
        print(f"         y range in east arm = "
              f"[{metrics['y_in_east_min']:.2f}, {metrics['y_in_east_max']:.2f}] m")

    # Check 3: does not mount kerb (y > -7) during turn
    if math.isnan(metrics["min_y_during_turn"]):
        print(f"\n  [SKIP] Kerb check: no turning phase recorded")
    else:
        ok3 = metrics["min_y_during_turn"] >= LANE_Y_MIN
        if not ok3:
            all_pass = False
        print(f"\n  [{'PASS' if ok3 else 'FAIL'}] Stays within road during turn "
              f"(y ≥ {LANE_Y_MIN} m)")
        print(f"         min y during turn = {metrics['min_y_during_turn']:.2f} m")

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
        description="Scenario 2: Right-turn junction traversal (south → east)")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open the MuJoCo viewer during simulation")
    args = parser.parse_args()

    print("=" * 70)
    print("  Scenario 2 — Right Turn at Junction (South → East)")
    print("=" * 70)
    print(f"  Scene:        {SCENE_PATH}")
    print(f"  Model:        {MODEL_PATH}")
    print(f"  Spawn:        x={NORTHBOUND_X} m, y=-35 m, heading=north")
    print(f"  Turn radius:  {TURN_RADIUS_M} m  "
          f"(delta={math.degrees(TURN_DELTA_RAD):.1f}°, ctrl={STEER_RIGHT_CTRL:.4f})")
    print(f"  Viewer:       {'on' if args.viewer else 'off (pass --viewer to enable)'}")

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
    print(f"  {'PASS' if ok else 'FAIL'} — right-turn scenario")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
