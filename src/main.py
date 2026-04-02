import mujoco
import mujoco.viewer
import numpy as np
import time

scene_spec = mujoco.MjSpec.from_file("../scenes/junction.xml")

robot_spec = mujoco.MjSpec.from_file("../models/simple_car.xml")

scene_spec.attach(robot_spec, frame="world", prefix="robot-")

paused = False

m = scene_spec.compile()
d  = mujoco.MjData(m)

vel_id     = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "robot-velocimeter")
acc_id     = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "robot-accelerometer")
chassis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,   "robot-chassis")
fl_jnt_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,  "robot-wheel_fl_steering")
fr_jnt_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,  "robot-wheel_fr_steering")
vel_adr    = int(m.sensor_adr[vel_id])
acc_adr    = int(m.sensor_adr[acc_id])
fl_qpos_adr = int(m.jnt_qposadr[fl_jnt_id])
fr_qpos_adr = int(m.jnt_qposadr[fr_jnt_id])

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
        f"a={accel:.2f} m/s2  "
        f"FL={fl_deg:.1f}deg  FR={fr_deg:.1f}deg"
    )
    viewer.user_scn.ngeom = 1

    viewer.sync()
    time.sleep(m.opt.timestep)

def key_callback(keycode):
    print(f"Key pressed: {keycode}")
    if keycode == 265:  # Use up arrow key to increase throttle
        d.ctrl[1] += 0.1  # Increase throttle control signal
    elif keycode == 264:  # Use down arrow key to decrease throttle
        d.ctrl[1] -= 0.1  # Decrease throttle control signal
    elif keycode == 262:  # Use right arrow key to increase steering angle
        # pass
        d.ctrl[0] += 1  # Increase steering control signal
    elif keycode == 263:  # Use left arrow key to decrease steering angle
        # pass
        d.ctrl[0] -= 1  # Decrease steering control signal
    print(f"Control signals: throttle={d.ctrl[1]}, steering={d.ctrl[0]}")

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
