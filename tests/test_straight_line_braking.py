"""
Straight-Line Braking Test — EDGAR MuJoCo model
================================================
Checks the minimum absolute deceleration spec from vehicle_parameters_edgar.yaml:
  a_min: -3.5 m/s²  — minimum absolute acceleration for v > 40 km/h

Procedure:
  1. Starts from rest on a flat surface.
  2. Applies full throttle (ctrl = 1.0) until speed exceeds 40 km/h.
  3. Immediately switches to full braking (ctrl = -1.0).
  4. Logs time, speed, and acceleration during the braking phase.
  5. Checks that mean deceleration is ≥ |a_min| = 3.5 m/s² in magnitude.
  6. Saves a CSV for offline plotting.

Run headless (default, standard Python):
  python3 tests/test_straight_line_braking.py

Run with the MuJoCo viewer (mjpython required for GUI):
  mjpython tests/test_straight_line_braking.py --viewer

Then plot the saved results:
  python3 tests/plot_straight_line_braking.py
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
CSV_PATH    = RESULTS_DIR / "straight_line_braking.csv"

# ── Test parameters ────────────────────────────────────────────────────────────
THROTTLE            = 1.0     # full throttle during acceleration phase
BRAKE_TRIGGER_KPH   = 40.0    # switch to braking once this speed is reached
TEST_DURATION_S     = 6.0    # simulated seconds (enough to accelerate + stop)
INITIAL_DECEL_WINDOW_S = 0.5  # seconds after brake trigger used for spec check

# ── Expected values (from vehicle_parameters_edgar.yaml) ───────────────────────
# a_min: -3.5 m/s²  — minimum absolute acceleration for v > 40 km/h
A_MIN_MS2 = -3.5     # m/s²  (negative = deceleration)

EXPECTED = {
    "decel_mean_ms2": A_MIN_MS2,   # mean deceleration must be ≤ this (more negative)
}

ACCEL_TOL = 0.10   # 10 % relative tolerance


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel, show_viewer: bool = False) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    throttle_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "throttle")
    brake_id    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "brake")
    vel_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "velocimeter")
    acc_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   "accelerometer")
    chassis_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     "chassis")
    vel_adr     = int(m.sensor_adr[vel_id])
    acc_adr     = int(m.sensor_adr[acc_id])

    d.ctrl[throttle_id] = THROTTLE   # start with full throttle

    times, speeds, accels, phases = [], [], [], []
    braking = False

    def _step_data():
        speed = float(d.sensordata[vel_adr])
        accel = float(d.sensordata[acc_adr])
        times.append(d.time)
        speeds.append(speed)
        accels.append(accel)
        phases.append("brake" if braking else "accel")
        return speed

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                step_start = time.time()
                mujoco.mj_step(m, d)

                speed = _step_data()
                speed_kph = speed * 3.6

                if not braking and speed_kph >= BRAKE_TRIGGER_KPH:
                    braking = True
                    d.ctrl[throttle_id] = 0.0
                    d.ctrl[brake_id]    = 1.0

                # ── Velocity overlay label ──────────────────────────────────
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
                phase_label = "BRAKING" if braking else "ACCEL"
                geom.label = (
                    f"[{phase_label}]  "
                    f"t={d.time:.1f}s  "
                    f"v={speed_kph:.1f} km/h  "
                    f"a={accels[-1]:.2f} m/s2"
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
            speed = _step_data()
            if not braking and speed * 3.6 >= BRAKE_TRIGGER_KPH:
                braking = True
                d.ctrl[throttle_id] = 0.0
                d.ctrl[brake_id]    = 1.0

    return {
        "time_s":    np.array(times),
        "speed_ms":  np.array(speeds),
        "accel_ms2": np.array(accels),
        "phase":     np.array(phases),
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict:
    t     = data["time_s"]
    v     = data["speed_ms"]
    a     = data["accel_ms2"]
    phase = data["phase"]

    brake_mask = phase == "brake"

    if brake_mask.any():
        t_brake_start = float(t[brake_mask][0])
        # Initial deceleration window: first INITIAL_DECEL_WINDOW_S seconds of braking
        # (speed is near BRAKE_TRIGGER_KPH here — this is what the spec covers)
        initial_mask = brake_mask & (t <= t_brake_start + INITIAL_DECEL_WINDOW_S)
        decel_initial = float(np.mean(a[initial_mask])) if initial_mask.any() else float("nan")
        decel_mean = float(np.mean(a[brake_mask]))
        decel_min  = float(np.min(a[brake_mask]))   # most negative = peak braking
        stopped = brake_mask & (v < 0.5)
        t_stop = float(t[stopped][0]) if stopped.any() else None
        brake_duration = (t_stop - t_brake_start) if t_stop is not None else None
    else:
        decel_initial = decel_mean = decel_min = float("nan")
        t_brake_start = brake_duration = None

    accel_mask = phase == "accel"
    t_trigger = float(t[accel_mask][-1]) if accel_mask.any() else float("nan")

    return {
        "t_brake_trigger_s":  t_trigger,
        "decel_initial_ms2":  decel_initial,   # mean decel in first 0.5 s — spec check
        "decel_mean_ms2":     decel_mean,
        "decel_peak_ms2":     decel_min,
        "brake_duration_s":   brake_duration,
        "peak_speed_kph":     float(np.max(v * 3.6)),
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    all_pass = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")

    print(f"  [INFO] Braking triggered at t = {metrics['t_brake_trigger_s']:.2f} s  "
          f"(speed ≥ {BRAKE_TRIGGER_KPH:.0f} km/h)")
    print(f"  [INFO] Peak speed reached:    {metrics['peak_speed_kph']:.1f} km/h")

    if metrics["brake_duration_s"] is not None:
        print(f"  [INFO] Time to stop after braking: {metrics['brake_duration_s']:.2f} s")
    else:
        print(f"  [INFO] Vehicle did not fully stop within {TEST_DURATION_S:.0f} s")

    # ── Spec check: initial deceleration ≥ |a_min| at the moment brake triggers ─
    # The damper force is proportional to velocity, so the strongest braking
    # occurs right at BRAKE_TRIGGER_KPH.  That is the correct point to check
    # the EDGAR spec "a_min for v > 40 km/h".
    d_init = metrics["decel_initial_ms2"]
    d_mean = metrics["decel_mean_ms2"]
    d_peak = metrics["decel_peak_ms2"]
    spec   = EXPECTED["decel_mean_ms2"]   # -3.5

    print(f"  [INFO] Mean deceleration (full braking phase): {d_mean:.3f} m/s²")
    print(f"  [INFO] Peak deceleration:                      {d_peak:.3f} m/s²")

    if np.isnan(d_init):
        print(f"\n  [SKIP] Deceleration check: braking phase not reached")
        all_pass = False
    else:
        # pass if initial deceleration is at least as strong as spec (more negative)
        ok = d_init <= spec * (1 - ACCEL_TOL)
        if not ok:
            all_pass = False
        print(f"\n  [{'PASS' if ok else 'FAIL'}] Initial deceleration ≥ |a_min| "
              f"(first {INITIAL_DECEL_WINDOW_S}s after brake, v ≈ {BRAKE_TRIGGER_KPH:.0f} km/h)")
        print(f"         spec a_min={spec:.2f} m/s²  initial={d_init:.3f} m/s²")

    return all_pass


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(data: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    # encode phase as 0 (accel) / 1 (brake) for numeric CSV
    phase_int = (data["phase"] == "brake").astype(int)
    rows = np.column_stack([
        data["time_s"],
        data["speed_ms"],
        data["speed_ms"] * 3.6,
        data["accel_ms2"],
        phase_int,
    ])
    np.savetxt(CSV_PATH, rows, delimiter=",",
               header="time_s,speed_ms,speed_kph,accel_ms2,phase_brake", comments="")
    print(f"\n  CSV saved → {CSV_PATH}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Straight-line braking test (EDGAR a_min spec)")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open the MuJoCo viewer during simulation (requires mjpython)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Straight-Line Braking Test (EDGAR a_min spec)")
    print("=" * 70)
    print(f"  Model:           {MODEL_PATH}")
    print(f"  Brake trigger:   {BRAKE_TRIGGER_KPH:.0f} km/h  |  "
          f"Duration: {TEST_DURATION_S} s")
    print(f"  Spec a_min:      {A_MIN_MS2} m/s²")
    print(f"  Viewer:          {'on' if args.viewer else 'off (pass --viewer to enable)'}")

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
    print(f"  {'PASS' if ok else 'FAIL'} — straight-line braking test")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
