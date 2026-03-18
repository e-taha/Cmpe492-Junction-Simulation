"""
Straight-Line Acceleration Test — EDGAR MuJoCo model
=====================================================
Procedure loosely based on ISO 15037-1 (general conditions for passenger car
handling & stability measurements).

What this test does:
  1. Starts from rest on a flat surface.
  2. Applies full throttle (ctrl = 1.0) instantaneously.
  3. Runs for a fixed duration with a live viewer, logging time, speed,
     and acceleration.
  4. Extracts key metrics: initial acceleration, time-to-speed milestones,
     and peak speed reached within the test window.
  5. Saves a CSV for offline plotting.

Run with the MuJoCo viewer (mjpython required for GUI):
  mjpython tests/test_straight_line_acceleration.py

Then plot the saved results with standard Python:
  python3 tests/plot_straight_line_acceleration.py
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent.parent / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "straight_line_acceleration.csv"

# ── Test parameters ────────────────────────────────────────────────────────────
THROTTLE        = 1.0    # full throttle  (ctrl range −1 … 1)
TEST_DURATION_S = 30.0   # simulated seconds to run

SPEED_TARGETS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # km/h

# ── Expected values ────────────────────────────────────────────────────────────
# Leave as None to skip the assertion; set a value for pass/fail.
EXPECTED = {
    # Analytically derived initial acceleration:
    #   torque/wheel = gear(1000) × coef(0.25) = 250 N·m
    #   F_drive/wheel = 250 / r_w(0.346) ≈ 722.5 N  →  total = 2890 N
    #   a_initial = 2890 / 2680 ≈ 1.08 m/s²
    "initial_accel_ms2": 1.08,
    "time_to_50kph_s":   None,   # fill from EDGAR specs when known
    "time_to_100kph_s":  None,
}

ACCEL_TOL = 0.10   # 10 % relative tolerance


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    throttle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "throttle")
    vel_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "velocimeter")
    acc_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "accelerometer")
    chassis_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     "chassis")
    vel_adr     = int(m.sensor_adr[vel_id])
    acc_adr     = int(m.sensor_adr[acc_id])

    d.ctrl[throttle_id] = THROTTLE

    times, speeds, accels = [], [], []

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and d.time < TEST_DURATION_S:
            step_start = time.time()
            mujoco.mj_step(m, d)

            speed = float(d.sensordata[vel_adr])
            accel = float(d.sensordata[acc_adr])
            times.append(d.time)
            speeds.append(speed)
            accels.append(accel)

            # ── Velocity overlay label (follows the car in world space) ────────
            viewer.user_scn.ngeom = 0
            geom = viewer.user_scn.geoms[0]
            # Place label 3 m above the chassis centre
            label_pos = d.xpos[chassis_id].copy()
            label_pos[2] += 3.0
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3),
                label_pos,
                np.eye(3).flatten(),
                np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),   # yellow
            )
            geom.label = (
                f"t={d.time:.1f}s  "
                f"v={speed * 3.6:.1f} km/h  "
                f"a={accel:.2f} m/s2"
            )
            viewer.user_scn.ngeom = 1
            # ───────────────────────────────────────────────────────────────────

            viewer.sync()
            elapsed = time.time() - step_start
            sleep_t = m.opt.timestep - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    return {
        "time_s":    np.array(times),
        "speed_ms":  np.array(speeds),
        "accel_ms2": np.array(accels),
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict:
    t     = data["time_s"]
    v     = data["speed_ms"]
    a     = data["accel_ms2"]
    v_kph = v * 3.6

    early_mask    = t <= 0.5
    initial_accel = float(np.mean(a[early_mask])) if early_mask.any() else float("nan")

    time_to = {}
    for target in SPEED_TARGETS:
        idx = np.argmax(v_kph >= target)
        time_to[target] = float(t[idx]) if v_kph[idx] >= target else None

    return {
        "initial_accel_ms2": initial_accel,
        "time_to_kph":       time_to,
        "peak_speed_kph":    float(np.max(v_kph)),
        "peak_speed_ms":     float(np.max(v)),
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    all_pass = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")

    a0    = metrics["initial_accel_ms2"]
    a_exp = EXPECTED["initial_accel_ms2"]
    if a_exp is not None:
        err = abs(a0 - a_exp) / (abs(a_exp) + 1e-12)
        ok  = err <= ACCEL_TOL
        if not ok:
            all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] Initial acceleration (avg first 0.5 s)")
        print(f"         expected={a_exp:.3f} m/s²  actual={a0:.3f} m/s²  "
              f"err={err*100:.1f}%")
    else:
        print(f"  [INFO] Initial acceleration (avg first 0.5 s): {a0:.3f} m/s²")

    print(f"\n  Time-to-speed milestones (full throttle from rest):")
    print(f"  {'Speed (km/h)':>14}  {'Time (s)':>10}  {'Expected (s)':>14}  status")
    print("  " + "─" * 52)

    for kph, t_val in metrics["time_to_kph"].items():
        exp = EXPECTED.get(f"time_to_{kph}kph_s")
        if t_val is None:
            print(f"  {kph:>14}  {'not reached':>10}  {'N/A':>14}  --")
            continue
        if exp is not None:
            err    = abs(t_val - exp) / (abs(exp) + 1e-12)
            ok     = err <= ACCEL_TOL
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            exp_str = f"{exp:.2f}"
        else:
            status  = "INFO"
            exp_str = "N/A"
        print(f"  {kph:>14}  {t_val:>10.2f}  {exp_str:>14}  {status}")

    print(f"\n  [INFO] Peak speed in {TEST_DURATION_S:.0f} s: "
          f"{metrics['peak_speed_kph']:.1f} km/h  "
          f"({metrics['peak_speed_ms']:.2f} m/s)")
    return all_pass


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(data: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([
        data["time_s"],
        data["speed_ms"],
        data["speed_ms"] * 3.6,
        data["accel_ms2"],
    ])
    np.savetxt(CSV_PATH, rows, delimiter=",",
               header="time_s,speed_ms,speed_kph,accel_ms2", comments="")
    print(f"\n  CSV saved → {CSV_PATH}")
    print(f"  To plot:   python3 tests/plot_straight_line_acceleration.py")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  Straight-Line Acceleration Test (ISO 15037)")
    print("=" * 70)
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Throttle: {THROTTLE}  |  Duration: {TEST_DURATION_S} s")

    if not MODEL_PATH.exists():
        print(f"\n  ERROR: model file not found at {MODEL_PATH}")
        sys.exit(1)

    spec  = mujoco.MjSpec.from_file(str(MODEL_PATH))
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([0.0, 0.0, 0.05])
    m = spec.compile()

    print("\n  Running simulation (close the viewer window to finish early)...")
    data    = run_simulation(m)
    metrics = extract_metrics(data)

    ok = print_report(metrics)
    save_csv(data)

    print("\n" + "=" * 70)
    print(f"  {'PASS' if ok else 'FAIL'} — straight-line acceleration test")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
