"""
Plot — Steering Limits Results
================================
Reads the CSV produced by test_steering_limits.py and saves a plot.

Run with standard Python (NOT mjpython — avoids threading conflicts):
  python3 tests/plot_steering_limits.py
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths & constants ───────────────────────────────────────────────────────────
RESULTS_DIR   = Path(__file__).parent / "results"
CSV_PATH      = RESULTS_DIR / "steering_limits.csv"
PNG_PATH      = RESULTS_DIR / "steering_limits.png"

DELTA_MAX_RAD = 0.610865
DELTA_MAX_DEG = DELTA_MAX_RAD * (180.0 / math.pi)
WHEELBASE_M   = 3.128
R_THEORY      = WHEELBASE_M / math.tan(DELTA_MAX_RAD)   # ≈ 4.47 m


def load_csv(path: Path) -> dict:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {
        "time_s":          data[:, 0],
        "x_m":             data[:, 1],
        "y_m":             data[:, 2],
        "speed_ms":        data[:, 3],
        "speed_kph":       data[:, 4],
        "accel_lon_ms2":   data[:, 5],
        "accel_lat_ms2":   data[:, 6],
        "yaw_rate_rads":   data[:, 7],
        "steer_angle_rad": data[:, 8],
    }


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}")
        print("Run the test first:  python3 tests/test_steering_limits.py")
        raise SystemExit(1)

    data = load_csv(CSV_PATH)
    t    = data["time_s"]
    x    = data["x_m"]
    y    = data["y_m"]
    v    = data["speed_kph"]
    a_lo = data["accel_lon_ms2"]
    a_la = data["accel_lat_ms2"]
    steer_deg = np.degrees(data["steer_angle_rad"])
    mean_speed = float(np.mean(data["speed_ms"][len(t) // 3:]))

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(
        f"Steering Limits — EDGAR MuJoCo Model\n"
        f"delta_max = {DELTA_MAX_RAD:.6f} rad ({DELTA_MAX_DEG:.2f}°),  "
        f"throttle = 0.15,  {t[-1]:.0f} s",
        fontsize=12,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel 1: XY trajectory ──────────────────────────────────────────────────
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xy.plot(y, x, color="steelblue", linewidth=1.2, label="Trajectory")
    ax_xy.plot(y[0], x[0], "go", markersize=6, label="Start")
    ax_xy.plot(y[-1], x[-1], "rs", markersize=6, label="End")

    # Overlay theoretical circle
    theta = np.linspace(0, 2 * math.pi, 300)
    # Estimate circle centre from data (mean of coords after settling)
    settle = len(t) // 3
    cx = float(np.mean(x[settle:]))
    cy = float(np.mean(y[settle:]))
    ax_xy.plot(cy + R_THEORY * np.sin(theta),
               cx + R_THEORY * np.cos(theta),
               color="orange", linestyle="--", linewidth=1.0,
               label=f"Theory R={R_THEORY:.2f} m")

    ax_xy.set_xlabel("Y (m)")
    ax_xy.set_ylabel("X (m)")
    ax_xy.set_title("XY Trajectory (top-down)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, linestyle="--", alpha=0.5)
    ax_xy.legend(fontsize=8)

    # ── Panel 2: Speed vs time ──────────────────────────────────────────────────
    ax_v = fig.add_subplot(gs[0, 1])
    ax_v.plot(t, v, color="steelblue", linewidth=1.2)
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Speed (km/h)")
    ax_v.set_title("Speed vs Time")
    ax_v.grid(True, linestyle="--", alpha=0.5)

    # ── Panel 3: Acceleration vs time ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[1, 0])
    ax_a.plot(t, a_la, color="tomato",   linewidth=1.0, label="Lateral (body-y)")
    ax_a.plot(t, a_lo, color="steelblue", linewidth=1.0, label="Longitudinal (body-x)")
    ax_a.axhline(0, color="black", linewidth=0.5)

    # Annotate theoretical lateral accel at mean speed
    a_lat_theory = mean_speed ** 2 / R_THEORY
    ax_a.axhline(a_lat_theory, color="orange", linestyle="--", linewidth=1.0,
                 label=f"Theory a_lat = v²/R = {a_lat_theory:.2f} m/s²")

    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Acceleration (m/s²)")
    ax_a.set_title("Body-Frame Acceleration vs Time")
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=8)

    # ── Panel 4: Steering angle vs time ────────────────────────────────────────
    ax_s = fig.add_subplot(gs[1, 1])
    ax_s.plot(t, steer_deg, color="purple", linewidth=1.0, label="Actual steer angle")
    ax_s.axhline(DELTA_MAX_DEG, color="red", linestyle="--", linewidth=1.0,
                 label=f"Commanded {DELTA_MAX_DEG:.2f}°")
    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylabel("Steering angle (°)")
    ax_s.set_title("Steering Angle vs Time")
    ax_s.grid(True, linestyle="--", alpha=0.5)
    ax_s.legend(fontsize=8)

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=150)
    print(f"Plot saved → {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
