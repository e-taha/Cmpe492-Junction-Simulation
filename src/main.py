import mujoco
import mujoco.viewer
import numpy as np
import time

scene_spec = mujoco.MjSpec.from_file("../scenes/empty_floor.xml")

robot_spec = mujoco.MjSpec.from_file("../models/simple_car.xml")

scene_spec.attach(robot_spec, frame="world", prefix="robot-")

model = scene_spec.compile()
data  = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Apply forward drive to both rear wheels
        # data.ctrl[0] = 0.8  # rl_spin
        # data.ctrl[1] = 0.8  # rr_spin
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)