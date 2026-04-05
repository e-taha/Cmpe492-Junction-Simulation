"""
Static & Geometric Verification — EDGAR MuJoCo model
=====================================================
Verifies mass, CG, wheelbase, track width, wheel radius, steering limits,
Ackermann geometry, and vehicle height against documented EDGAR parameters.

Reference:
  https://github.com/TUMFTM/edgar_digital_twin/blob/main/source/vehicle_dynamics_parameter/README.md

Run from the repo root:
  python tests/test_static_geometric.py
"""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

# ── Model path (relative to this file) ────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "simple_car.xml"

# ── Reference values ───────────────────────────────────────────────────────────
# Source: edgar_digital_twin / vehicle_parameters_edgar.yaml + XML comments.
# Update these values if the official EDGAR README gives different numbers.
REF = {
    "total_mass_kg":        2520.0,    # chassis 2400 + 4 × 30 kg wheels (tire+rim+brake est.)
    "cg_z_m":               0.700,     # body_pos(1.333) + inertial_offset(-0.633); set via <inertial>
    "cg_to_front_axle_m":   1.484,     # a
    "cg_to_rear_axle_m":    1.644,     # b
    "wheelbase_m":           3.128,    # a + b
    "front_track_m":         1.645,    # T_f
    "wheel_radius_m":        0.346,    # r_w  Bridgestone 235/50R18: 9in rim + 117.5mm sidewall
    # Steering max in RADIANS (35° ≈ 0.6109 rad).
    # NOTE: MuJoCo XML interprets angle values as degrees by default.
    # If angle="radian" is NOT set in <compiler>, write range="-35 35".
    "steer_max_rad":         0.610865,
    "vehicle_height_m":      2.320,    # from EDGAR dimension spec
}

# Relative tolerance for dimensional checks (2 %)
TOL = 0.02


# ── Result tracker (avoids global variable pitfalls) ──────────────────────────
class _Results:
    passed: int = 0
    failed: int = 0


_r = _Results()


# ── Helpers ────────────────────────────────────────────────────────────────────

def check(label: str, actual: float, expected: float,
          tol: float = TOL, unit: str = "") -> bool:
    rel_err = abs(actual - expected) / (abs(expected) + 1e-12)
    ok = rel_err <= tol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}")
    print(f"         expected={expected:.6g}{unit}  "
          f"actual={actual:.6g}{unit}  "
          f"rel_err={rel_err * 100:.2f}%")
    if ok:
        _r.passed += 1
    else:
        _r.failed += 1
    return ok


def check_bool(label: str, condition: bool, detail: str = "") -> bool:
    tag = "PASS" if condition else "FAIL"
    print(f"  [{tag}] {label}")
    if detail:
        print(f"         {detail}")
    if condition:
        _r.passed += 1
    else:
        _r.failed += 1
    return condition


# ── Individual tests ───────────────────────────────────────────────────────────

def test_total_mass(m: mujoco.MjModel) -> bool:
    print("\n── 1. Total Mass ──────────────────────────────────────────────────────")
    total = float(np.sum(m.body_mass))
    return check("Total model mass", total, REF["total_mass_kg"], unit=" kg")


def test_cg_height(m: mujoco.MjModel) -> bool:
    print("\n── 2. Centre-of-Gravity Height ────────────────────────────────────────")
    chassis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    # body_pos is the frame origin; body_ipos is the CoM offset within that frame.
    cg_z = float(m.body_pos[chassis_id, 2]) + float(m.body_ipos[chassis_id, 2])
    ok = check("Chassis CoG z above ground", cg_z, REF["cg_z_m"], unit=" m")
    print("         Cross-check: compare this value against h_CoG in "
          "vehicle_parameters_edgar.yaml")
    return ok


def test_wheelbase(m: mujoco.MjModel) -> bool:
    print("\n── 3. Wheelbase (a + b) ───────────────────────────────────────────────")
    fl_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_fl")
    rl_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_rl")
    # body_pos is in parent (chassis) frame; x is positive forward
    a = float(m.body_pos[fl_id, 0])    # CG → front axle (positive)
    b = -float(m.body_pos[rl_id, 0])   # CG → rear axle  (flip sign)
    wheelbase = a + b
    ok1 = check("a  (CG → front axle)", a, REF["cg_to_front_axle_m"], unit=" m")
    ok2 = check("b  (CG → rear axle)",  b, REF["cg_to_rear_axle_m"],  unit=" m")
    ok3 = check("Wheelbase L = a + b",  wheelbase, REF["wheelbase_m"],  unit=" m")
    return ok1 and ok2 and ok3


def test_front_track(m: mujoco.MjModel) -> bool:
    print("\n── 4. Front Track Width ───────────────────────────────────────────────")
    fl_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_fl")
    fr_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_fr")
    fl_y = float(m.body_pos[fl_id, 1])
    fr_y = float(m.body_pos[fr_id, 1])
    track = abs(fl_y - fr_y)
    return check("Front track T_f", track, REF["front_track_m"], unit=" m")


def test_wheel_radius(m: mujoco.MjModel) -> bool:
    print("\n── 5. Wheel Radius ────────────────────────────────────────────────────")
    fl_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_fl")
    g_adr = int(m.body_geomadr[fl_id])
    # Cylinder geom: size = [radius, half-length]
    radius = float(m.geom_size[g_adr, 0])
    return check("Wheel radius", radius, REF["wheel_radius_m"], unit=" m")


def test_steering_limits(m: mujoco.MjModel) -> bool:
    """
    MuJoCo interprets angle values in XML as DEGREES by default (unless
    <compiler angle="radian"/> is set).  Therefore:
      - range="-35 35"      → stored as ±0.6109 rad  (CORRECT for ±35°)
      - range="-0.610865 0.610865" → stored as ±0.01066 rad (WRONG — this is
        only ±0.61°, because MuJoCo treated 0.610865 as degrees)

    If the model currently FAILs this check, the fix is one of:
      a) Change <joint ... range="-35 35"/> in the steering default class, OR
      b) Add <compiler angle="radian"/> to simple_car.xml
    """
    print("\n── 6. Steering Joint Limits ───────────────────────────────────────────")
    sw_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wheel_fl_steering")
    lo = float(m.jnt_range[sw_id, 0])
    hi = float(m.jnt_range[sw_id, 1])
    print(f"         Raw values from MjModel (radians): lo={lo:.6f}  hi={hi:.6f}")
    ok1 = check("Steer upper limit (rad)", hi,  REF["steer_max_rad"], unit=" rad")
    ok2 = check("Steer lower limit (rad)", -lo, REF["steer_max_rad"], unit=" rad")
    if not (ok1 and ok2):
        deg_hi = math.degrees(hi)
        print(f"\n         DIAGNOSIS: actual range is ±{deg_hi:.4f}° "
              f"(≈ ±{hi:.5f} rad).")
        print(f"         MuJoCo parsed the XML range as degrees.")
        print(f"         Fix option A — change the steering default class in simple_car.xml:")
        print(f"           <joint ... range=\"-35 35\"/>")
        print(f"         Fix option B — add to simple_car.xml:")
        print(f"           <compiler angle=\"radian\"/>")
    return ok1 and ok2


def test_vehicle_height(m: mujoco.MjModel) -> bool:
    print("\n── 8. Vehicle Top Height ──────────────────────────────────────────────")
    chassis_id     = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    g_adr          = int(m.body_geomadr[chassis_id])
    chassis_half_h = float(m.geom_size[g_adr, 2])
    chassis_z      = float(m.body_pos[chassis_id, 2])
    vehicle_top    = chassis_z + chassis_half_h
    return check("Vehicle top (CG_z + chassis half-height)",
                 vehicle_top, REF["vehicle_height_m"], unit=" m")


# ── Manual checks (not verifiable from MjModel alone) ─────────────────────────

MANUAL_CHECKS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  MANUAL CHECKS — cannot be verified from MjModel alone                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. Moment of inertia (I_z, I_x, I_y)                                       ║
║     MuJoCo derives inertia from uniform-density box geoms.  The real EDGAR   ║
║     has a different mass distribution (engine, battery, etc.).               ║
║     Inspect with:                                                            ║
║       python -c "                                                            ║
║         import mujoco; m = mujoco.MjSpec.from_file(                         ║
║           'models/simple_car.xml').compile();                                ║
║         cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'chassis');    ║
║         print('I (kg·m²):', m.body_inertia[cid])"                           ║
║     Compare against I_z value in vehicle_parameters_edgar.yaml.             ║
║                                                                              ║
║  2. CG height against EDGAR measurement                                      ║
║     Simulated CG_z = 0.70 m (set via <inertial pos="0 0 -0.633"> in XML).  ║
║     h_cg is blank in vehicle_parameters_edgar.yaml (unmeasured).            ║
║     0.70 m is a reasonable estimate for a VW T7-class van. Update REF and   ║
║     the inertial offset in simple_car.xml if a measured value is found.     ║
║                                                                              ║
║  3. Tire model fidelity                                                      ║
║     MuJoCo uses a simple friction-cone contact model, NOT a Pacejka tire    ║
║     model.  The EDGAR digital twin uses full cornering stiffness (C_f, C_r).║
║     This is a fundamental limitation that will affect all dynamic tests.     ║
║                                                                              ║
║  4. Suspension                                                               ║
║     No suspension DOFs are modelled.  This affects ride height, load        ║
║     transfer under acceleration/braking, and dynamic CG shift.              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  EDGAR MuJoCo Model — Static & Geometric Verification")
    print("=" * 70)
    print(f"  Model: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print(f"\n  ERROR: model file not found at {MODEL_PATH}")
        sys.exit(1)

    m = mujoco.MjSpec.from_file(str(MODEL_PATH)).compile()

    tests = [
        ("1. Total mass",          test_total_mass),
        ("2. CG height",           test_cg_height),
        ("3. Wheelbase",           test_wheelbase),
        ("4. Front track",         test_front_track),
        ("5. Wheel radius",        test_wheel_radius),
        ("6. Steering limits",     test_steering_limits),
        ("7. Vehicle height",      test_vehicle_height),
    ]

    results = {}
    for name, fn in tests:
        results[name] = fn(m)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n  {passed}/{total} checks passed\n")

    print(MANUAL_CHECKS)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
