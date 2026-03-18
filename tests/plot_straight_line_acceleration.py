"""
Plot — Straight-Line Acceleration Results
=========================================
Reads the CSV produced by test_straight_line_acceleration.py and saves a plot.

Run with standard Python (NOT mjpython — avoids threading conflicts):
  python3 tests/plot_straight_line_acceleration.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "straight_line_acceleration.csv"
PNG_PATH    = RESULTS_DIR / "straight_line_acceleration.png"

SPEED_TARGETS = [20, 40, 60, 80, 100]   # km/h milestones to annotate


def load_csv(path: Path) -> dict:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {
        "time_s":    data[:, 0],
        "speed_ms":  data[:, 1],
        "speed_kph": data[:, 2],
        "accel_ms2": data[:, 3],
    }


def time_to_speed(t, v_kph, target_kph):
    idx = np.argmax(v_kph >= target_kph)
    return float(t[idx]) if v_kph[idx] >= target_kph else None


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}")
        print("Run the test first:  mjpython tests/test_straight_line_acceleration.py")
        raise SystemExit(1)

    data  = load_csv(CSV_PATH)
    t     = data["time_s"]
    v_kph = data["speed_kph"]
    a     = data["accel_ms2"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        "Straight-Line Acceleration — EDGAR MuJoCo Model\n"
        f"(ISO 15037, full throttle from rest, {t[-1]:.0f} s)",
        fontsize=12,
    )

    # ── Speed plot ─────────────────────────────────────────────────────────────
    ax1.plot(t, v_kph, color="steelblue", linewidth=1.5, label="Speed")
    ax1.set_ylabel("Speed (km/h)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    for target in SPEED_TARGETS:
        t_val = time_to_speed(t, v_kph, target)
        if t_val is not None:
            ax1.axhline(target, color="gray", linestyle=":", linewidth=0.8)
            ax1.plot(t_val, target, "o", color="steelblue", markersize=4)
            ax1.annotate(
                f"{target} km/h @ {t_val:.1f} s",
                xy=(t_val, target),
                xytext=(t_val + 0.3, target + 1.5),
                fontsize=7,
                color="dimgray",
            )

    ax1.legend(loc="upper left", fontsize=9)

    # ── Acceleration plot ───────────────────────────────────────────────────────
    ax2.plot(t, a, color="tomato", linewidth=1.0, label="Forward accel (body frame)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=150)
    print(f"Plot saved → {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
