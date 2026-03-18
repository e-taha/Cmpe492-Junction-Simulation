"""
Straight-Line Acceleration Test — EDGAR MuJoCo model
=====================================================
Procedure loosely based on ISO 15037-1 (general conditions for passenger car
handling & stability measurements).

What this test does:
  1. Starts from rest on a flat surface.
  2. Applies full throttle (ctrl = 1.0) instantaneously.
  3. Runs for a fixed duration, logging time, speed, and acceleration.
  4. Extracts key metrics: initial acceleration, time-to-speed milestones,
     and peak speed reached within the test window.
  5. Saves a CSV for offline plotting.

Run headless (default, standard Python):
  python3 tests/test_straight_line_acceleration.py

Run with the MuJoCo viewer (mjpython required for GUI):
  mjpython tests/test_straight_line_acceleration.py --viewer

Then plot the saved results with standard Python:
  python3 tests/plot_straight_line_acceleration.py
"""

import argparse
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

# ── Expected values (from vehicle_parameters_edgar.yaml) ───────────────────────
# a_max: 2.5 m/s²  — maximum acceleration for v > 40 km/h (full throttle)
# At full throttle the model must sustain at least this value in that speed band.
#
# Analytically derived initial acceleration (gear=2180, coef=0.25, r_w=0.346, m=2520):
#   torque/wheel = 2180 × 0.25 = 545 N·m
#   F_drive/wheel = 545 / 0.346 ≈ 1575 N  →  total = 6300 N
#   a_initial = 6300 / 2520 ≈ 2.50 m/s²
A_MAX_MS2        = 2.5          # m/s²  — spec limit for v > 40 km/h
A_MAX_SPEED_BAND = (40.0, 100.0)  # km/h  — speed range where spec applies

EXPECTED = {
    "accel_40_100kph_min_ms2": A_MAX_MS2,   # must be ≥ this throughout the band
    "time_to_50kph_s":        None,  # fill from EDGAR specs when known
    "time_to_100kph_s":       None,
}

ACCEL_TOL = 0.10   # 10 % relative tolerance


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel, show_viewer: bool = False) -> dict:
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

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                step_start = time.time()
                mujoco.mj_step(m, d)

                speed = float(d.sensordata[vel_adr])
                accel = float(d.sensordata[acc_adr])
                times.append(d.time)
                speeds.append(speed)
                accels.append(accel)

                # ── Velocity overlay label (follows the car in world space) ────
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
                    f"v={speed * 3.6:.1f} km/h  "
                    f"a={accel:.2f} m/s2"
                )
                viewer.user_scn.ngeom = 1
                # ─────────────────────────────────────────────────────────────

                viewer.sync()
                elapsed = time.time() - step_start
                sleep_t = m.opt.timestep - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            mujoco.mj_step(m, d)
            times.append(d.time)
            speeds.append(float(d.sensordata[vel_adr]))
            accels.append(float(d.sensordata[acc_adr]))

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

    lo, hi = A_MAX_SPEED_BAND
    band_mask = (v_kph >= lo) & (v_kph <= hi)
    accel_band_min = float(np.min(a[band_mask])) if band_mask.any() else float("nan")
    accel_band_mean = float(np.mean(a[band_mask])) if band_mask.any() else float("nan")

    time_to = {}
    for target in SPEED_TARGETS:
        idx = np.argmax(v_kph >= target)
        time_to[target] = float(t[idx]) if v_kph[idx] >= target else None

    return {
        "initial_accel_ms2":       initial_accel,
        "accel_40_100kph_min_ms2": accel_band_min,
        "accel_40_100kph_mean_ms2": accel_band_mean,
        "time_to_kph":             time_to,
        "peak_speed_kph":          float(np.max(v_kph)),
        "peak_speed_ms":           float(np.max(v)),
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    all_pass = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")

    # ── Initial acceleration — informational only (no EDGAR spec below 40 km/h) ─
    a0 = metrics["initial_accel_ms2"]
    print(f"  [INFO] Initial acceleration (avg first 0.5 s): {a0:.3f} m/s²")

    # ── Spec check: acceleration ≥ a_max (2.5 m/s²) in 40–100 km/h band ───────
    a_band_min  = metrics["accel_40_100kph_min_ms2"]
    a_band_mean = metrics["accel_40_100kph_mean_ms2"]
    a_spec      = EXPECTED["accel_40_100kph_min_ms2"]
    lo, hi      = A_MAX_SPEED_BAND
    if np.isnan(a_band_min):
        print(f"\n  [SKIP] Acceleration in {lo:.0f}–{hi:.0f} km/h band: "
              f"speed band not reached")
    else:
        ok = a_band_min >= a_spec * (1 - ACCEL_TOL)
        if not ok:
            all_pass = False
        print(f"\n  [{'PASS' if ok else 'FAIL'}] Acceleration in "
              f"{lo:.0f}–{hi:.0f} km/h band (full throttle, from EDGAR spec)")
        print(f"         spec a_max={a_spec:.2f} m/s²  "
              f"min={a_band_min:.3f} m/s²  mean={a_band_mean:.3f} m/s²")

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
    parser = argparse.ArgumentParser(description="Straight-line acceleration test (ISO 15037)")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open the MuJoCo viewer during simulation (requires mjpython)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Straight-Line Acceleration Test (ISO 15037)")
    print("=" * 70)
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Throttle: {THROTTLE}  |  Duration: {TEST_DURATION_S} s")
    print(f"  Viewer:   {'on' if args.viewer else 'off (pass --viewer to enable)'}")

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
        print("\n  Running simulation (close the viewer window to finish early)...")
    else:
        print("\n  Running simulation (headless)...")
    data    = run_simulation(m, show_viewer=args.viewer)
    metrics = extract_metrics(data)

    ok = print_report(metrics)
    save_csv(data)

    print("\n" + "=" * 70)
    print(f"  {'PASS' if ok else 'FAIL'} — straight-line acceleration test")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
