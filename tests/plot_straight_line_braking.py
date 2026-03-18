"""
Plot — Straight-Line Braking Results
=====================================
Reads the CSV produced by test_straight_line_braking.py and saves a plot.

Run with standard Python (NOT mjpython — avoids threading conflicts):
  python3 tests/plot_straight_line_braking.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "straight_line_braking.csv"
PNG_PATH    = RESULTS_DIR / "straight_line_braking.png"

A_MIN_MS2         = -3.5    # EDGAR spec
BRAKE_TRIGGER_KPH = 40.0


def load_csv(path: Path) -> dict:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {
        "time_s":      data[:, 0],
        "speed_ms":    data[:, 1],
        "speed_kph":   data[:, 2],
        "accel_ms2":   data[:, 3],
        "phase_brake": data[:, 4].astype(bool),   # 0=accel, 1=brake
    }


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}")
        print("Run the test first:  python3 tests/test_straight_line_braking.py")
        raise SystemExit(1)

    data       = load_csv(CSV_PATH)
    t          = data["time_s"]
    v_kph      = data["speed_kph"]
    a          = data["accel_ms2"]
    is_braking = data["phase_brake"]

    # Find brake trigger time and peak speed
    t_brake     = float(t[is_braking][0]) if is_braking.any() else None
    peak_speed  = float(np.max(v_kph))
    mean_decel  = float(np.mean(a[is_braking])) if is_braking.any() else float("nan")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        "Straight-Line Braking — EDGAR MuJoCo Model\n"
        f"(full throttle → full brake at {BRAKE_TRIGGER_KPH:.0f} km/h, "
        f"spec a_min = {A_MIN_MS2} m/s²)",
        fontsize=12,
    )

    # ── Speed plot ──────────────────────────────────────────────────────────────
    ax1.plot(t[~is_braking], v_kph[~is_braking],
             color="steelblue", linewidth=1.5, label="Acceleration phase")
    ax1.plot(t[is_braking],  v_kph[is_braking],
             color="tomato",   linewidth=1.5, label="Braking phase")
    ax1.set_ylabel("Speed (km/h)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    if t_brake is not None:
        ax1.axvline(t_brake, color="orange", linestyle="--", linewidth=1.0,
                    label=f"Brake applied @ {t_brake:.2f} s")
        ax1.annotate(
            f"Brake at\n{BRAKE_TRIGGER_KPH:.0f} km/h",
            xy=(t_brake, BRAKE_TRIGGER_KPH),
            xytext=(t_brake + 0.3, BRAKE_TRIGGER_KPH + 5),
            fontsize=7, color="darkorange",
        )

    ax1.plot([], [], " ", label=f"Peak speed: {peak_speed:.1f} km/h")
    ax1.legend(loc="upper right", fontsize=8)

    # ── Acceleration plot ───────────────────────────────────────────────────────
    ax2.plot(t[~is_braking], a[~is_braking],
             color="steelblue", linewidth=1.0, label="Acceleration phase")
    ax2.plot(t[is_braking],  a[is_braking],
             color="tomato",   linewidth=1.0, label="Braking phase")

    ax2.axhline(0,        color="black",  linewidth=0.5)
    ax2.axhline(A_MIN_MS2, color="red",   linewidth=1.0, linestyle="--",
                label=f"Spec a_min = {A_MIN_MS2} m/s²")

    if not np.isnan(mean_decel):
        ax2.axhline(mean_decel, color="tomato", linewidth=1.0, linestyle=":",
                    label=f"Mean decel = {mean_decel:.2f} m/s²")

    if t_brake is not None:
        ax2.axvline(t_brake, color="orange", linestyle="--", linewidth=1.0)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=150)
    print(f"Plot saved → {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
