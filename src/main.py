import mujoco
import mujoco.viewer
import numpy as np
import time

scene_spec = mujoco.MjSpec.from_file("../scenes/empty_floor.xml")

robot_spec = mujoco.MjSpec.from_file("../models/simple_car.xml")

scene_spec.attach(robot_spec, frame="world", prefix="robot-")

paused = False

m = scene_spec.compile()
d  = mujoco.MjData(m)

# Pressing SPACE key toggles the paused state.
def mujoco_viewer_callback(keycode):
    global paused
    if keycode == ord(' '):  # Use ord(' ') for space key comparison
        paused = not paused

def step():
    mujoco.mj_step(m, d)
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
