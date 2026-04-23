"""
Scenario 3 — Highway Merge
===========================
Three vehicles follow reference paths using the Pure Pursuit controller:

  v_right   — main road, eastbound right lane (y = -1.75)
               straight path from x=-15 to x=65

  v_left    — main road, eastbound left lane (y = -5.25)
               straight path from x=-25 to x=65

  v_merging — on-ramp then merges into the right lane
               path: ramp segment → smooth blend arc → eastbound right lane

Merge path geometry
-------------------
  Ramp segment   : (-2, -15) → (20, -7)  at 20° from east
  Blend arc      : entry (20, -7) heading 20°, exit (≈50, -1.75) heading east
                   arc radius R = 5.25 / (1 - cos 20°) ≈ 87 m
                   (chosen so the arc lifts the vehicle exactly from y=-7 to y=-1.75)
  Exit straight  : (≈50, -1.75) → (65, -1.75) eastbound

Path colours (viewer)
---------------------
  v_right   : blue   (0.3, 0.5, 1.0)
  v_left    : purple (0.7, 0.3, 1.0)
  v_merging : green  (0.0, 0.9, 0.2)

Usage
-----
  mjpython scenarios/merge/index.py
  mjpython scenarios/merge/index.py --viewer
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

# ── Module path ────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from pure_pursuit import (PurePursuit, straight_segment, arc_segment,
                          make_path, heading_from_xmat, draw_path_geoms)

# ── File paths ─────────────────────────────────────────────────────────────────
SCENE_PATH  = ROOT_DIR / "scenes" / "merge.xml"
MODEL_PATH  = ROOT_DIR / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"

# ── Vehicle / road constants ───────────────────────────────────────────────────
WHEELBASE_M      = 3.128
LOOKAHEAD_M      = 5.0
RAMP_ANGLE_RAD   = math.radians(20.0)

# ── Reference paths ────────────────────────────────────────────────────────────
#
#  v_right — straight east along y = -1.75
#
PATH_RIGHT = straight_segment(-15.0, -1.75, 65.0, -1.75)

#
#  v_left — straight east along y = -5.25
#
PATH_LEFT = straight_segment(-25.0, -5.25, 65.0, -5.25)

#
#  v_merging — ramp + blend arc + exit
#
#  Blend arc derivation:
#    Entry (20, -7) heading 20° → right turn → exit heading east (0°)
#    Arc centre = entry + R * (sin θ, -cos θ)  where θ = ramp angle
#    We need exit y = -1.75:  R*(1 - cos θ) = 7 - 1.75 = 5.25
#
_R   = 5.25 / (1.0 - math.cos(RAMP_ANGLE_RAD))                 # ≈ 87.06 m
_CX  = 20.0 + _R * math.sin(RAMP_ANGLE_RAD)                    # ≈ 49.79 m
_CY  = -7.0 - _R * math.cos(RAMP_ANGLE_RAD)                    # ≈ -88.82 m
_T0  = math.atan2(-7.0 - _CY, 20.0 - _CX)                     # ≈ 1.920 rad  (110°)
_T1  = math.pi / 2                                              # 90° — exits pointing east
_EXIT_X = _CX                                                   # ≈ 49.79 m (arc exit x)

PATH_MERGING = make_path(
    straight_segment(-2.0, -15.0, 20.0, -7.0),                 # ramp
    arc_segment(_CX, _CY, _R, _T0, _T1),                       # blend (CW, θ decreasing)
    straight_segment(_EXIT_X, -1.75, 65.0, -1.75),             # exit east
)

# Path colours: [R, G, B, A]
PATH_COLORS = {
    "v_right":   [0.3, 0.5, 1.0, 0.9],   # blue
    "v_left":    [0.7, 0.3, 1.0, 0.9],   # purple
    "v_merging": [0.0, 0.9, 0.2, 0.9],   # green
}
PATHS = {
    "v_right":   PATH_RIGHT,
    "v_left":    PATH_LEFT,
    "v_merging": PATH_MERGING,
}

# ── Scenario parameters ────────────────────────────────────────────────────────
THROTTLE        = 0.18
TEST_DURATION_S = 30.0


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> mujoco.MjModel:
    scene_spec   = mujoco.MjSpec.from_file(str(SCENE_PATH))
    spec_right   = mujoco.MjSpec.from_file(str(MODEL_PATH))
    spec_left    = mujoco.MjSpec.from_file(str(MODEL_PATH))
    spec_merging = mujoco.MjSpec.from_file(str(MODEL_PATH))
    scene_spec.attach(spec_right,   frame="spawn_main_right", prefix="v_right-")
    scene_spec.attach(spec_left,    frame="spawn_main_left",  prefix="v_left-")
    scene_spec.attach(spec_merging, frame="spawn_merging",    prefix="v_merging-")
    return scene_spec.compile()


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel, show_viewer: bool = False) -> None:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    # ── IDs for all three vehicles ─────────────────────────────────────────────
    def _ids(prefix):
        return {
            "steering":  mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}steering"),
            "throttle":  mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}throttle"),
            "brake":     mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}brake"),
            "chassis":   mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     f"{prefix}chassis"),
        }

    ids = {
        "v_right":   _ids("v_right-"),
        "v_left":    _ids("v_left-"),
        "v_merging": _ids("v_merging-"),
    }

    trackers = {
        "v_right":   PurePursuit(PATH_RIGHT,   WHEELBASE_M, LOOKAHEAD_M),
        "v_left":    PurePursuit(PATH_LEFT,    WHEELBASE_M, LOOKAHEAD_M),
        "v_merging": PurePursuit(PATH_MERGING, WHEELBASE_M, LOOKAHEAD_M),
    }

    def step():
        for name, vid in ids.items():
            cx  = float(d.xpos[vid["chassis"]][0])
            cy  = float(d.xpos[vid["chassis"]][1])
            hdg = heading_from_xmat(d.xmat, vid["chassis"])
            d.ctrl[vid["steering"]] = trackers[name].compute_ctrl(cx, cy, hdg)
            d.ctrl[vid["throttle"]] = THROTTLE
            d.ctrl[vid["brake"]]    = 0.0
        mujoco.mj_step(m, d)

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                t0 = time.time()
                step()

                # ── Draw all 3 paths + lookahead spheres + labels ──────────────
                geom_idx = 0
                for name, vid in ids.items():
                    cx  = float(d.xpos[vid["chassis"]][0])
                    cy  = float(d.xpos[vid["chassis"]][1])
                    hdg = heading_from_xmat(d.xmat, vid["chassis"])
                    lp  = trackers[name].lookahead_point(cx, cy)

                    lbl_pos     = d.xpos[vid["chassis"]].copy()
                    lbl_pos[2] += 3.0

                    geom_idx = draw_path_geoms(
                        viewer, PATHS[name],
                        color        = PATH_COLORS[name],
                        lookahead_xy = lp,
                        label_pos    = lbl_pos,
                        label_text   = (
                            f"{name}  "
                            f"({cx:.1f}, {cy:.1f})  "
                            f"hdg={math.degrees(hdg):.0f}°"
                        ),
                        start_idx    = geom_idx,
                    )
                viewer.user_scn.ngeom = geom_idx

                viewer.sync()
                elapsed = time.time() - t0
                sleep_t = m.opt.timestep - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            step()
        print(f"  Simulated {TEST_DURATION_S} s.")
        for name, vid in ids.items():
            cx = float(d.xpos[vid["chassis"]][0])
            cy = float(d.xpos[vid["chassis"]][1])
            print(f"  {name}: final pos ({cx:.2f}, {cy:.2f})")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scenario 3: Highway merge with Pure Pursuit (3 vehicles)")
    parser.add_argument("--viewer", action="store_true", default=False,
                        help="Open the MuJoCo viewer")
    args = parser.parse_args()

    print("=" * 70)
    print("  Scenario 3 — Highway Merge")
    print("=" * 70)
    print(f"  Scene:      {SCENE_PATH}")
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Lookahead:  {LOOKAHEAD_M} m")
    print(f"  Throttle:   {THROTTLE}")
    print("  Paths:")
    print(f"    v_right   {len(PATH_RIGHT)} pts — straight east y=-1.75")
    print(f"    v_left    {len(PATH_LEFT)} pts — straight east y=-5.25")
    print(f"    v_merging {len(PATH_MERGING)} pts — ramp + arc (R={_R:.1f} m) + exit")
    print(f"  Viewer:     {'on' if args.viewer else 'off (pass --viewer to enable)'}")

    for p, label in [(SCENE_PATH, "scene"), (MODEL_PATH, "model")]:
        if not p.exists():
            print(f"\n  ERROR: {label} not found: {p}")
            sys.exit(1)

    m = build_model()
    print("\n  Running simulation...")
    run_simulation(m, show_viewer=args.viewer)


if __name__ == "__main__":
    main()
