# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a vehicle dynamics simulation for the EDGAR autonomous vehicle, implemented using the **MuJoCo** physics engine. The project validates vehicle behavior (steering, acceleration, braking) against documented specifications defined in `vehicle_parameters_edgar.yaml`.

## Commands

### Run Interactive Simulation

```bash
mjpython src/main.py
```

Controls: arrow keys (steer/throttle), space bar (pause). Displays real-time speed, acceleration overlay.

### Run Tests (headless)

```bash
python tests/test_static_geometric.py
python tests/test_steering_limits.py
python tests/test_straight_line_acceleration.py
python tests/test_straight_line_braking.py
```

### Run Tests with MuJoCo Viewer (requires `mjpython`)

```bash
mjpython tests/test_steering_limits.py --viewer
mjpython tests/test_straight_line_acceleration.py --viewer
mjpython tests/test_straight_line_braking.py --viewer
```

### Generate Plots (after running tests)

```bash
python tests/plot_steering_limits.py
python tests/plot_straight_line_acceleration.py
python tests/plot_straight_line_braking.py
```

Test results (CSV + PNG) are saved to `tests/results/`.

### Dependencies

```bash
pip install mujoco numpy matplotlib
```

## Architecture

### Key Files

- `vehicle_parameters_edgar.yaml` — Ground truth vehicle spec (mass, dimensions, steering limits, acceleration/braking bounds)
- `models/simple_car.xml` — MuJoCo model: chassis, 4 wheels, actuators, sensors, Ackermann steering constraints
- `scenes/empty_floor.xml` — Flat ground scene included by tests/main
- `src/main.py` — Interactive viewer simulation
- `tests/test_*.py` — Validation tests against spec
- `tests/plot_*.py` — Post-process CSVs into matplotlib figures

### MuJoCo Model Design

**Steering**: A virtual steering wheel (position actuator, ±35°) is coupled to the actual front wheels (FL/FR) via polynomial equality constraints, not implementing Ackermann geometry.

**Throttle/Brake**: A tendon system links all 4 wheel spin joints equally. Throttle uses a torque-based motor (gear=2520); braking uses a velocity-proportional damper (`kv=95`), so braking force = `-kv × ctrl × velocity`.

**Sensors**: IMU site provides gyroscope, velocimeter, and accelerometer readings.

**Simulation timestep**: 5ms (200Hz).

### Test Pattern

All tests follow the same structure:

1. Load model from XML
2. Run simulation for a fixed duration with specific control inputs
3. Log telemetry (position, velocity, acceleration, steering angle) each step
4. Assert metrics against spec with tolerances (typically 5–15%)
5. Save results as CSV, then a separate `plot_*.py` script renders PNGs

### Vehicle Specification (from `vehicle_parameters_edgar.yaml`)

- Total mass: 2520 kg (chassis 2400 kg + 4 × 30 kg wheels)
- Wheelbase: 3.128 m, CG height: 1.333 m
- Max steering angle: ±0.610865 rad (±35°)
- Max acceleration: ≥ 2.5 m/s²; max deceleration: ≥ 3.5 m/s² (for v > 40 km/h)
