import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

scene_spec = mujoco.MjSpec.from_file(str(ROOT / "scenes/junction.xml"))

robot_spec = mujoco.MjSpec.from_file(str(ROOT / "models/simple_car.xml"))

scene_spec.attach(robot_spec, frame="world", prefix="robot-")

paused = False

m = scene_spec.compile()
d  = mujoco.MjData(m)

vel_id     = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "robot-velocimeter")
acc_id     = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "robot-accelerometer")
chassis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,   "robot-chassis")
sw_ten_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON,  "robot-steering_link")
fl_jnt_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,  "robot-wheel_fl_steering")
fr_jnt_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,  "robot-wheel_fr_steering")
vel_adr    = int(m.sensor_adr[vel_id])
acc_adr    = int(m.sensor_adr[acc_id])
fl_qpos_adr = int(m.jnt_qposadr[fl_jnt_id])
fr_qpos_adr = int(m.jnt_qposadr[fr_jnt_id])

# Mesh offset tuning: find the chassis_visual geom and track its offset
mesh_geom_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot-chassis_visual")
mesh_offset    = m.geom_pos[mesh_geom_id].copy()  # starts at whatever is in the model
OFFSET_STEP    = 0.05  # metres per keypress


# Pressing SPACE key toggles the paused state.
def mujoco_viewer_callback(keycode):
    global paused
    if keycode == ord(' '):  # Use ord(' ') for space key comparison
        paused = not paused

def step():
    mujoco.mj_step(m, d)

    speed   = float(d.sensordata[vel_adr])
    accel   = float(d.sensordata[acc_adr])
    fl_deg  = float(np.degrees(d.qpos[fl_qpos_adr]))
    fr_deg  = float(np.degrees(d.qpos[fr_qpos_adr]))
    steering_link_rad = float(d.ten_length[sw_ten_id])
    steering_link_deg = float(np.degrees(steering_link_rad))

    viewer.user_scn.ngeom = 0
    label_pos = d.xpos[chassis_id].copy()

    geom0 = viewer.user_scn.geoms[0]
    pos0 = label_pos.copy(); pos0[2] += 3.5
    mujoco.mjv_initGeom(geom0, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                        pos0, np.eye(3).flatten(),
                        np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32))
    geom0.label = (f"t={d.time:.1f}s v={speed*3.6:.1f}km/h a={accel:.2f}m/s2"
                   f" FL={fl_deg:.1f} FR={fr_deg:.1f} St={steering_link_deg:.1f}deg")

    ox, oy, oz = mesh_offset
    geom1 = viewer.user_scn.geoms[1]
    pos1 = label_pos.copy(); pos1[2] += 2.8
    mujoco.mjv_initGeom(geom1, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                        pos1, np.eye(3).flatten(),
                        np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32))
    geom1.label = f"mesh x={ox:.3f} y={oy:.3f} z={oz:.3f} [W/S=X A/D=Y Q/E=Z]"

    viewer.user_scn.ngeom = 2

    viewer.sync()
    time.sleep(m.opt.timestep)

def key_callback(keycode):
    global mesh_offset
    print(f"Key pressed: {keycode}")
    if keycode == 265:  # up arrow → throttle up
        d.ctrl[1] += 0.1
    elif keycode == 264:  # down arrow → throttle down
        d.ctrl[1] -= 0.1
    elif keycode == 262:  # right arrow → steer right
        d.ctrl[0] += np.radians(10)
    elif keycode == 263:  # left arrow → steer left
        d.ctrl[0] -= np.radians(10)
    # Mesh offset tuning (W/S = X, A/D = Y, Q/E = Z)
    elif keycode == ord('W'):
        mesh_offset[0] += OFFSET_STEP
    elif keycode == ord('S'):
        mesh_offset[0] -= OFFSET_STEP
    elif keycode == ord('D'):
        mesh_offset[1] += OFFSET_STEP
    elif keycode == ord('A'):
        mesh_offset[1] -= OFFSET_STEP
    elif keycode == ord('E'):
        mesh_offset[2] += OFFSET_STEP
    elif keycode == ord('Q'):
        mesh_offset[2] -= OFFSET_STEP
    else:
        print(f"Control signals: throttle={d.ctrl[1]}, steering={d.ctrl[0]}")
        return
    # Apply offset to model geom so the viewer picks it up immediately
    m.geom_pos[mesh_geom_id] = mesh_offset
    print(f"mesh_offset → x={mesh_offset[0]:.3f}  y={mesh_offset[1]:.3f}  z={mesh_offset[2]:.3f}")

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    while viewer.is_running():
        step_counter = 0        
        step_start = time.time()
        # car_velocity = d.sensor("robot-velocimeter").data[0]
        # car_acceleration = d.sensor("robot-accelerometer").data[0]      
        if not paused:
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            start_time = d.time
            while viewer.is_running() and d.time - start_time < 30:
                step()
