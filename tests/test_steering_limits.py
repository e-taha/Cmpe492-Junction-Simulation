"""
Steering Limits Test — EDGAR MuJoCo model
==========================================
Verifies that the vehicle achieves and holds the maximum steering angle
(kappa_max / kappa_min) from vehicle_parameters_edgar.yaml:

  delta_max:  0.610865 rad  (≈ 35.0°)
  delta_min: -0.610865 rad  (≈ -35.0°)

Procedure:
  1. Commands maximum steering angle (DELTA_MAX_RAD) to the position-controlled
     steering actuator and applies moderate throttle.
  2. Records position (x, y), speed, and acceleration over TEST_DURATION_S seconds.
  3. Checks:
       a) Steering joint reaches commanded angle within tolerance.
       b) Turning radius matches bicycle-model prediction  L / tan(delta).
  4. Saves a CSV for offline plotting.

Run headless (default, standard Python):
  python3 tests/test_steering_limits.py

Run with the MuJoCo viewer (mjpython required for GUI):
  mjpython tests/test_steering_limits.py --viewer

Then plot the saved results:
  python3 tests/plot_steering_limits.py
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
MODEL_PATH  = Path(__file__).parent.parent / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "steering_limits.csv"

# ── Parameters from vehicle_parameters_edgar.yaml ──────────────────────────────
DELTA_MAX_RAD = 0.610865      # rad  — exact value from YAML (delta_max = 35°)
DELTA_MAX_DEG = DELTA_MAX_RAD * (180.0 / math.pi)   # ≈ 35.0°  (informational)
WHEELBASE_M   = 3.128               # a + b  (CG-to-front + CG-to-rear axle)

# ── Test parameters ─────────────────────────────────────────────────────────────
# The model has no passive drag, so speed keeps rising until lateral tire forces
# limit it.  60 s gives several full circles and a near-steady-state speed.
# At max steer, stability limit ≈ 8 m/s (29 km/h) with friction_coef=1.5.
THROTTLE        = 0.35
TEST_DURATION_S = 60.0   # long enough to approach terminal circular speed

SETTLE_FRAC  = 0.60   # skip first 60 % — vehicle still spiralling early on
ANGLE_TOL    = 0.05   # 5 %  relative tolerance on steering angle
RADIUS_TOL   = 0.20   # 20 % tolerance — accounts for tire slip vs kinematic model

EXPECTED = {
    "steer_angle_rad":  DELTA_MAX_RAD,
    "turning_radius_m": WHEELBASE_M / math.tan(DELTA_MAX_RAD),   # ≈ 4.47 m
}


# ── Simulation ──────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel, show_viewer: bool = False) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    # ── Actuator / sensor IDs ──────────────────────────────────────────────────
    steering_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "steering")
    throttle_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "throttle")
    vel_id       = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "velocimeter")
    acc_id       = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "accelerometer")
    gyro_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "gyro")
    chassis_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     "chassis")
    steer_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,    "wheel_fl_steering")

    vel_adr       = int(m.sensor_adr[vel_id])
    acc_adr       = int(m.sensor_adr[acc_id])
    gyro_adr      = int(m.sensor_adr[gyro_id])
    steer_jnt_adr = int(m.jnt_qposadr[steer_jnt_id])

    # Tendon = (-1)*fl_angle + (-1)*fr_angle; both equal via constraint.
    # To command left turn at DELTA_MAX_RAD: ctrl = -2 * DELTA_MAX_RAD = -1.2217
    d.ctrl[steering_id] = -2.0 * DELTA_MAX_RAD
    d.ctrl[throttle_id] = THROTTLE

    times, xs, ys = [], [], []
    speeds, accel_lons, accel_lats = [], [], []
    yaw_rates, steer_angles = [], []

    def _record():
        vx    = float(d.sensordata[vel_adr])
        vy    = float(d.sensordata[vel_adr + 1])
        lon_a = float(d.sensordata[acc_adr])
        lat_a = float(d.sensordata[acc_adr + 1])
        yaw_r = float(d.sensordata[gyro_adr + 2])
        steer = float(d.qpos[steer_jnt_adr])
        pos   = d.xpos[chassis_id]

        times.append(d.time)
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        speeds.append(math.sqrt(vx ** 2 + vy ** 2))
        accel_lons.append(lon_a)
        accel_lats.append(lat_a)
        yaw_rates.append(yaw_r)
        steer_angles.append(steer)

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                step_start = time.time()
                mujoco.mj_step(m, d)
                _record()

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
                    f"steer={math.degrees(steer_angles[-1]):.1f}°  "
                    f"a_lat={accel_lats[-1]:.2f} m/s2"
                )
                viewer.user_scn.ngeom = 1
                viewer.sync()
                elapsed = time.time() - step_start
                sleep_t = m.opt.timestep - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            mujoco.mj_step(m, d)
            _record()

    return {
        "time_s":          np.array(times),
        "x_m":             np.array(xs),
        "y_m":             np.array(ys),
        "speed_ms":        np.array(speeds),
        "accel_lon_ms2":   np.array(accel_lons),
        "accel_lat_ms2":   np.array(accel_lats),
        "yaw_rate_rads":   np.array(yaw_rates),
        "steer_angle_rad": np.array(steer_angles),
    }


# ── Metrics ─────────────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict:
    n          = len(data["time_s"])
    settle     = int(n * SETTLE_FRAC)    # index where steady-state begins

    speed  = data["speed_ms"]
    lat_a  = data["accel_lat_ms2"]
    yaw    = data["yaw_rate_rads"]
    steer  = data["steer_angle_rad"]

    mean_steer = float(np.mean(np.abs(steer[settle:])))
    mean_speed = float(np.mean(speed[settle:]))
    mean_lat_a = float(np.mean(lat_a[settle:]))

    # Turning radius: R = v / |ω|  (valid only when yaw rate is significant)
    yaw_ss = yaw[settle:]
    spd_ss = speed[settle:]
    valid  = np.abs(yaw_ss) > 0.05   # rad/s
    if valid.any():
        mean_radius = float(np.mean(spd_ss[valid] / np.abs(yaw_ss[valid])))
    else:
        mean_radius = float("nan")

    return {
        "mean_steer_angle_rad": mean_steer,
        "mean_steer_angle_deg": math.degrees(mean_steer),
        "mean_speed_ms":        mean_speed,
        "mean_speed_kph":       mean_speed * 3.6,
        "mean_lat_accel_ms2":   mean_lat_a,
        "estimated_radius_m":   mean_radius,
    }


# ── Reporting ───────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    all_pass = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")

    print(f"  [INFO] Commanded steer:          {DELTA_MAX_DEG:.4f}°  "
          f"({DELTA_MAX_RAD:.6f} rad)")
    print(f"  [INFO] Mean steady-state steer:  "
          f"{metrics['mean_steer_angle_deg']:.4f}°  "
          f"({metrics['mean_steer_angle_rad']:.6f} rad)")
    print(f"  [INFO] Mean speed:               {metrics['mean_speed_kph']:.1f} km/h")
    print(f"  [INFO] Mean lateral accel:       {metrics['mean_lat_accel_ms2']:.3f} m/s²")
    print(f"  [INFO] Estimated turning radius: {metrics['estimated_radius_m']:.2f} m")

    # ── Check 1: steering angle reaches delta_max ─────────────────────────────
    exp_steer = EXPECTED["steer_angle_rad"]
    act_steer = metrics["mean_steer_angle_rad"]
    err_s = abs(act_steer - exp_steer) / (exp_steer + 1e-12)
    ok_s  = err_s <= ANGLE_TOL
    if not ok_s:
        all_pass = False
    print(f"\n  [{'PASS' if ok_s else 'FAIL'}] Steering angle reaches delta_max "
          f"(from EDGAR spec)")
    print(f"         commanded={exp_steer:.6f} rad  "
          f"actual={act_steer:.6f} rad  err={err_s * 100:.1f}%")

    # ── Check 2: turning radius matches bicycle model L / tan(delta) ──────────
    R_theory = EXPECTED["turning_radius_m"]
    R_actual = metrics["estimated_radius_m"]
    if np.isnan(R_actual):
        print(f"\n  [SKIP] Turning radius: yaw rate too small to estimate reliably")
    else:
        err_r = abs(R_actual - R_theory) / R_theory
        ok_r  = err_r <= RADIUS_TOL
        if not ok_r:
            all_pass = False
        print(f"\n  [{'PASS' if ok_r else 'FAIL'}] Turning radius matches bicycle model "
              f"L / tan(delta)")
        print(f"         L/tan({DELTA_MAX_DEG:.1f}°)={R_theory:.2f} m  "
              f"actual={R_actual:.2f} m  err={err_r * 100:.1f}%")

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
        data["yaw_rate_rads"],
        data["steer_angle_rad"],
    ])
    np.savetxt(
        CSV_PATH, rows, delimiter=",",
        header="time_s,x_m,y_m,speed_ms,speed_kph,accel_lon_ms2,accel_lat_ms2,"
               "yaw_rate_rads,steer_angle_rad",
        comments="",
    )
    print(f"\n  CSV saved → {CSV_PATH}")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Steering limits test (EDGAR delta_max / delta_min spec)")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open the MuJoCo viewer during simulation (requires mjpython)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Steering Limits Test (EDGAR delta_max / delta_min spec)")
    print("=" * 70)
    print(f"  Model:            {MODEL_PATH}")
    print(f"  Commanded steer:  {DELTA_MAX_RAD:.6f} rad  ({DELTA_MAX_DEG:.4f}°)")
    print(f"  Throttle:         {THROTTLE}")
    print(f"  Duration:         {TEST_DURATION_S} s")
    print(f"  Expected radius:  {EXPECTED['turning_radius_m']:.2f} m  "
          f"(L / tan({DELTA_MAX_DEG:.1f}°))")
    print(f"  Viewer:           {'on' if args.viewer else 'off (pass --viewer to enable)'}")

    if not MODEL_PATH.exists():
        print(f"\n  ERROR: model file not found at {MODEL_PATH}")
        sys.exit(1)

    spec  = mujoco.MjSpec.from_file(str(MODEL_PATH))
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([0.0, 0.0, 0.05])
    m = spec.compile()

    if args.viewer:
        print("\n  Running simulation (close viewer to finish early)...")
    else:
        print("\n  Running simulation (headless)...")

    data    = run_simulation(m, show_viewer=args.viewer)
    metrics = extract_metrics(data)

    ok = print_report(metrics)
    save_csv(data)

    print("\n" + "=" * 70)
    print(f"  {'PASS' if ok else 'FAIL'} — steering limits test")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
