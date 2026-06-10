"""
NCAP AEB C2C CCFhos — Car-to-Car Front Head-On Same-lane (2023)
===============================================================
Source scenario: NCAP_scenarios/NCAP_AEB_C2C_CCFhos_2023.xosc

Two Edgar vehicles approach head-on in the SAME lane (x=+1.75 m):

  Ego  (northbound, x=+1.75 m)
       Straight path from y=0 to y=350.  Speed-controlled to 70 km/h.

  GVT  (southbound, x=+1.75 m)
       Straight path south.  Speed-controlled to 70 km/h.
       Starts at y=311 m (= (v_ego + v_gvt) × startupTime = 38.888 × 8 s).

Physics collision between Ego and GVT is enabled via MuJoCo's contact solver.
An AABB overlap check records the first collision time and speed for reporting.

NCAP parameters (from .xosc)
-----------------------------
  Ego / GVT speed  : 70 km/h  (= 19.444 m/s)
  startupTime      : 8 s
  _GVT_initDist    : (v_ego + v_gvt) × 8 = 311.1 m

Derived timing
--------------
  Closing speed    : 38.888 m/s
  TTC at t=0       : 311.1 / 38.888 ≈ 8.0 s

Usage
-----
  mjpython scenarios/NCAP_AEB_C2C_CCFhos/index.py
  mjpython scenarios/NCAP_AEB_C2C_CCFhos/index.py --viewer
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from pure_pursuit import (PurePursuit, straight_segment,
                          make_path, heading_from_xmat, draw_path_geoms)
from props import add_roadside_props

SCENE_PATH  = ROOT_DIR / "scenes" / "ncap_straight_road_ccfhos.xml"
MODEL_PATH  = ROOT_DIR / "models" / "simple_car.xml"
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "ncap_ccfhos.csv"

# ── NCAP parameters ────────────────────────────────────────────────────────────
EGO_SPEED_KPH   = 70.0
EGO_SPEED_MS    = EGO_SPEED_KPH / 3.6          # ≈ 19.444 m/s
GVT_SPEED_MS    = EGO_SPEED_MS
STARTUP_TIME_S  = 8.0

GVT_INIT_DIST_M = (EGO_SPEED_MS + GVT_SPEED_MS) * STARTUP_TIME_S  # ≈ 311.1 m
EGO_START_Y     = 0.0
GVT_START_Y     = GVT_INIT_DIST_M

LANE_X          = 1.75                          # both vehicles in same lane

CLOSING_SPEED_MS     = EGO_SPEED_MS + GVT_SPEED_MS
TIME_TO_COLLISION_S  = GVT_INIT_DIST_M / CLOSING_SPEED_MS  # ≈ 8.0 s

# ── Vehicle constants ──────────────────────────────────────────────────────────
WHEELBASE_M     = 3.128
WHEEL_RADIUS_M  = 0.346
LOOKAHEAD_M     = 8.0
TEST_DURATION_S = 15.0

EGO_HALF_LEN    = 4.973 / 2
EGO_HALF_WID    = 1.941 / 2
GVT_HALF_LEN    = EGO_HALF_LEN
GVT_HALF_WID    = EGO_HALF_WID

COLOR_EGO = [0.3, 0.5, 1.0, 0.9]
COLOR_GVT = [1.0, 0.3, 0.2, 0.9]

PATH_EGO = straight_segment(LANE_X, -5.0,    LANE_X, 350.0)
PATH_GVT = straight_segment(LANE_X, GVT_START_Y, LANE_X, -50.0)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> mujoco.MjModel:
    scene_spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    spec_ego   = mujoco.MjSpec.from_file(str(MODEL_PATH))
    spec_gvt   = mujoco.MjSpec.from_file(str(MODEL_PATH))

    add_roadside_props(scene_spec, y_min=-10.0, y_max=320.0, seed=13)

    scene_spec.attach(spec_ego, frame="spawn_ego", prefix="ego-")
    scene_spec.attach(spec_gvt, frame="spawn_gvt", prefix="gvt-")

    m = scene_spec.compile()

    # Floor collides with all vehicles (contype=3 matches default vehicle contype=1).
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        m.geom_contype[floor_id]     = 3
        m.geom_conaffinity[floor_id] = 3

    return m


# ── Vehicle helpers ────────────────────────────────────────────────────────────

def _vehicle_ids(m: mujoco.MjModel, prefix: str) -> dict:
    def _aid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    def _sid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,   n)
    def _bid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,     n)
    def _jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,    n)
    return {
        "steering": _aid(f"{prefix}steering"),
        "throttle": _aid(f"{prefix}throttle"),
        "brake":    _aid(f"{prefix}brake"),
        "vel":      _sid(f"{prefix}velocimeter"),
        "acc":      _sid(f"{prefix}accelerometer"),
        "chassis":  _bid(f"{prefix}chassis"),
        "root":     _jid(f"{prefix}root"),
        "fl_spin":  _jid(f"{prefix}fl_spin"),
        "fr_spin":  _jid(f"{prefix}fr_spin"),
        "rl_spin":  _jid(f"{prefix}rl_spin"),
        "rr_spin":  _jid(f"{prefix}rr_spin"),
        "fl_steer": _jid(f"{prefix}wheel_fl_steering"),
    }


def _init_speed(m, d, ids, speed_ms, direction):
    root_dof = int(m.jnt_dofadr[ids["root"]])
    d.qvel[root_dof + 1] = speed_ms * direction
    omega = speed_ms / WHEEL_RADIUS_M
    for jn in ("fl_spin", "fr_spin", "rl_spin", "rr_spin"):
        d.qvel[int(m.jnt_dofadr[ids[jn]])] = omega


def _step_vehicle(m, d, ids, tracker, target_speed):
    chassis = ids["chassis"]
    cx  = float(d.xpos[chassis][0])
    cy  = float(d.xpos[chassis][1])
    hdg = heading_from_xmat(d.xmat, chassis)

    vel_adr = int(m.sensor_adr[ids["vel"]])
    vx  = float(d.sensordata[vel_adr])
    vy  = float(d.sensordata[vel_adr + 1])
    spd = math.sqrt(vx**2 + vy**2)

    speed_err = target_speed - spd
    throttle  = max(0.0, min(1.0, 0.15 + 0.6 * speed_err))

    d.ctrl[ids["steering"]] = tracker.compute_ctrl(cx, cy, hdg)
    d.ctrl[ids["throttle"]] = throttle
    d.ctrl[ids["brake"]]    = 0.0


def _check_collision(ego_x, ego_y, gvt_x, gvt_y):
    lon = abs(ego_y - gvt_y) < (EGO_HALF_LEN + GVT_HALF_LEN)
    lat = abs(ego_x - gvt_x) < (EGO_HALF_WID + GVT_HALF_WID)
    return lon and lat


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_simulation(m: mujoco.MjModel, show_viewer: bool = False,
                   show_labels: bool = True) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    ids_ego = _vehicle_ids(m, "ego-")
    ids_gvt = _vehicle_ids(m, "gvt-")

    _init_speed(m, d, ids_ego, EGO_SPEED_MS, +1.0)
    _init_speed(m, d, ids_gvt, GVT_SPEED_MS, -1.0)

    tracker_ego = PurePursuit(PATH_EGO, WHEELBASE_M, LOOKAHEAD_M)
    tracker_gvt = PurePursuit(PATH_GVT, WHEELBASE_M, LOOKAHEAD_M)

    times, ego_ys, gvt_ys  = [], [], []
    ego_speeds             = []
    accel_lons, accel_lats = [], []
    steer_angles           = []
    separations            = []
    collision_flags        = []

    collision_time  = None
    collision_speed = None

    ego_vel_adr = int(m.sensor_adr[ids_ego["vel"]])
    ego_acc_adr = int(m.sensor_adr[ids_ego["acc"]])
    ego_fl_qpos = int(m.jnt_qposadr[ids_ego["fl_steer"]])

    def _record(gvt_y, collided):
        pos = d.xpos[ids_ego["chassis"]]
        vx  = float(d.sensordata[ego_vel_adr])
        vy  = float(d.sensordata[ego_vel_adr + 1])
        spd = math.sqrt(vx**2 + vy**2)
        times.append(d.time)
        ego_ys.append(float(pos[1]))
        gvt_ys.append(gvt_y)
        ego_speeds.append(spd)
        accel_lons.append(float(d.sensordata[ego_acc_adr]))
        accel_lats.append(float(d.sensordata[ego_acc_adr + 1]))
        steer_angles.append(float(d.qpos[ego_fl_qpos]))
        separations.append(abs(float(pos[1]) - gvt_y))
        collision_flags.append(int(collided))

    def _sim_step():
        if collision_time is None:
            _step_vehicle(m, d, ids_ego, tracker_ego, EGO_SPEED_MS)
            _step_vehicle(m, d, ids_gvt, tracker_gvt, GVT_SPEED_MS)
        else:
            for ids_v in (ids_ego, ids_gvt):
                d.ctrl[ids_v["throttle"]] = 0.0
                d.ctrl[ids_v["brake"]]    = 0.0
                d.ctrl[ids_v["steering"]] = 0.0
        mujoco.mj_step(m, d)
        return float(d.xpos[ids_gvt["chassis"]][0]), float(d.xpos[ids_gvt["chassis"]][1])

    if show_viewer:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and d.time < TEST_DURATION_S:
                t0 = time.time()
                gvt_x, gvt_y = _sim_step()
                ego_pos = d.xpos[ids_ego["chassis"]]
                ex, ey  = float(ego_pos[0]), float(ego_pos[1])
                collided = _check_collision(ex, ey, gvt_x, gvt_y)
                _record(gvt_y, collided)
                if collided and collision_time is None:
                    collision_time  = d.time
                    collision_speed = ego_speeds[-1]

                geom_idx = 0
                if show_labels:
                    for ids_v, tracker, path, color, name in [
                        (ids_ego, tracker_ego, PATH_EGO, COLOR_EGO, "Ego"),
                        (ids_gvt, tracker_gvt, PATH_GVT, COLOR_GVT, "GVT"),
                    ]:
                        cx  = float(d.xpos[ids_v["chassis"]][0])
                        cy  = float(d.xpos[ids_v["chassis"]][1])
                        lp  = tracker.lookahead_point(cx, cy)
                        lbl = d.xpos[ids_v["chassis"]].copy(); lbl[2] += 3.5
                        va  = int(m.sensor_adr[ids_v["vel"]])
                        spd_kph = math.sqrt(float(d.sensordata[va])**2 +
                                            float(d.sensordata[va+1])**2) * 3.6
                        geom_idx = draw_path_geoms(
                            viewer, path, color=color,
                            lookahead_xy=lp, label_pos=lbl,
                            label_text=f"{name} {spd_kph:.0f}km/h",
                            start_idx=geom_idx,
                        )

                    gap  = gvt_y - ey
                    ttc  = gap / CLOSING_SPEED_MS if gap > 0 else 0.0
                    if geom_idx < viewer.user_scn.maxgeom:
                        import mujoco as _mj
                        lbl_pos = d.xpos[ids_ego["chassis"]].copy(); lbl_pos[2] += 6.0
                        g = viewer.user_scn.geoms[geom_idx]
                        _mj.mjv_initGeom(g, _mj.mjtGeom.mjGEOM_LABEL,
                                         np.zeros(3), np.asarray(lbl_pos),
                                         np.eye(3).flatten(),
                                         np.array([1.,1.,0.,1.], dtype=np.float32))
                        g.label = (f"gap={gap:.1f}m  TTC={ttc:.2f}s  "
                                   f"{'*** COLLISION ***' if collided else ''}")
                        geom_idx += 1

                viewer.user_scn.ngeom = geom_idx
                viewer.sync()
                sleep_t = m.opt.timestep - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
    else:
        while d.time < TEST_DURATION_S:
            gvt_x, gvt_y = _sim_step()
            ego_pos  = d.xpos[ids_ego["chassis"]]
            ex, ey   = float(ego_pos[0]), float(ego_pos[1])
            collided = _check_collision(ex, ey, gvt_x, gvt_y)
            _record(gvt_y, collided)
            if collided and collision_time is None:
                collision_time  = d.time
                collision_speed = ego_speeds[-1]

    return {
        "time_s":             np.array(times),
        "ego_y_m":            np.array(ego_ys),
        "gvt_y_m":            np.array(gvt_ys),
        "speed_ms":           np.array(ego_speeds),
        "accel_lon_ms2":      np.array(accel_lons),
        "accel_lat_ms2":      np.array(accel_lats),
        "steer_angle_rad":    np.array(steer_angles),
        "separation_m":       np.array(separations),
        "collision":          np.array(collision_flags),
        "collision_time_s":   collision_time,
        "collision_speed_ms": collision_speed,
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def extract_metrics(data: dict) -> dict:
    speeds = data["speed_ms"]
    return {
        "mean_ego_speed_kph":  float(np.mean(speeds)) * 3.6,
        "peak_ego_speed_kph":  float(np.max(speeds))  * 3.6,
        "collision_detected":  data["collision_time_s"] is not None,
        "collision_time_s":    data["collision_time_s"] or float("nan"),
        "collision_speed_kph": (data["collision_speed_ms"] * 3.6
                                if data["collision_speed_ms"] is not None
                                else float("nan")),
        "min_separation_m":    float(np.min(data["separation_m"])),
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> bool:
    ok = True
    print("\n── Metrics ────────────────────────────────────────────────────────────")
    print(f"  [INFO] Mean Ego speed:      {metrics['mean_ego_speed_kph']:.1f} km/h")
    print(f"  [INFO] Peak Ego speed:      {metrics['peak_ego_speed_kph']:.1f} km/h")
    print(f"  [INFO] Min separation:      {metrics['min_separation_m']:.2f} m")
    if not math.isnan(metrics["collision_time_s"]):
        print(f"  [INFO] Collision time:      {metrics['collision_time_s']:.3f} s "
              f"(expected ≈ {TIME_TO_COLLISION_S:.1f} s)")
        print(f"  [INFO] Ego speed at coll.:  {metrics['collision_speed_kph']:.1f} km/h")

    ok1 = 70.0 * 0.85 <= metrics["mean_ego_speed_kph"] <= 70.0 * 1.15
    ok  = ok and ok1
    print(f"\n  [{'PASS' if ok1 else 'FAIL'}] Ego maintains ~70 km/h "
          f"(mean = {metrics['mean_ego_speed_kph']:.1f} km/h)")

    ok2 = metrics["collision_detected"]
    ok  = ok and ok2
    print(f"\n  [{'PASS' if ok2 else 'FAIL'}] Collision detected "
          f"(expected without AEB — t={metrics['collision_time_s']:.2f} s, "
          f"v={metrics['collision_speed_kph']:.1f} km/h)")

    ttc_ok = not math.isnan(metrics["collision_time_s"]) and \
             abs(metrics["collision_time_s"] - TIME_TO_COLLISION_S) < 1.5
    ok = ok and ttc_ok
    print(f"\n  [{'PASS' if ttc_ok else 'FAIL'}] Collision time within 1.5 s of expected "
          f"({TIME_TO_COLLISION_S:.1f} s)")

    return ok


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([
        data["time_s"],
        data["ego_y_m"],    data["gvt_y_m"],
        data["speed_ms"],   data["speed_ms"] * 3.6,
        data["accel_lon_ms2"], data["accel_lat_ms2"],
        data["steer_angle_rad"],
        data["separation_m"],  data["collision"],
    ])
    np.savetxt(CSV_PATH, rows, delimiter=",",
               header="time_s,ego_y_m,gvt_y_m,ego_speed_ms,ego_speed_kph,"
                      "accel_lon_ms2,accel_lat_ms2,steer_angle_rad,"
                      "longitudinal_sep_m,collision",
               comments="")
    print(f"\n  CSV saved → {CSV_PATH}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NCAP AEB C2C CCFhos — head-on same-lane, 2 Edgar vehicles")
    parser.add_argument("--viewer",   action="store_true", default=False)
    parser.add_argument("--no-label", action="store_true", default=False,
                        help="Hide speed/TTC overlays in the viewer")
    args = parser.parse_args()

    print("=" * 70)
    print("  NCAP AEB C2C CCFhos — Car-to-Car Front Head-On Same-Lane")
    print("=" * 70)
    print(f"  Scene:           {SCENE_PATH}")
    print(f"  Ego speed:       {EGO_SPEED_KPH} km/h  (northbound, x={LANE_X} m)")
    print(f"  GVT speed:       {EGO_SPEED_KPH} km/h  (southbound, x={LANE_X} m)")
    print(f"  GVT start y:     {GVT_START_Y:.1f} m")
    print(f"  Closing speed:   {CLOSING_SPEED_MS*3.6:.1f} km/h")
    print(f"  Expected TTC:    {TIME_TO_COLLISION_S:.1f} s")
    print(f"  Viewer:          {'on' if args.viewer else 'off (pass --viewer)'}")

    for p, label in [(SCENE_PATH, "scene"), (MODEL_PATH, "model")]:
        if not p.exists():
            print(f"\n  ERROR: {label} not found: {p}")
            sys.exit(1)

    m = build_model()
    print("\n  Running simulation...")
    data    = run_simulation(m, show_viewer=args.viewer,
                             show_labels=not args.no_label)
    metrics = extract_metrics(data)
    ok      = print_report(metrics)
    save_csv(data)

    print("\n" + "=" * 70)
    print(f"  {'PASS' if ok else 'FAIL'} — NCAP AEB C2C CCFhos")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
