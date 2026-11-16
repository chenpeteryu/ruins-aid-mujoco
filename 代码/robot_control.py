"""Local flat-ground torque control and sensor feedback for the public Go2."""
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from basic_sim import actuator_addresses
from check_resources import RESOURCES

LEGS = ("FR", "FL", "RR", "RL")


def make_environment():
    if mujoco.__version__ != "3.17.0":
        raise RuntimeError("MuJoCo 3.17.0 required")
    source = RESOURCES / "robot/go2/physics/go2.xml"
    xml = ET.parse(source).getroot()
    xml.find("compiler").set("meshdir", str(source.parent / "assets"))
    option = xml.find("option")
    option.attrib.update(timestep="0.002", solver="Newton", iterations="50", ls_iterations="10", cone="elliptic", impratio="100")
    asset = xml.find("asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", width="512", height="512", rgb1="0.2 0.25 0.3", rgb2="0.5 0.55 0.6")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="20 20", texuniform="true")
    world = xml.find("worldbody")
    ET.SubElement(world, "geom", name="local_floor", type="plane", size="20 20 .1", material="grid")
    ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1", diffuse=".9 .9 .9")
    # Non-colliding visual landmarks make camera motion observable on a flat test pad.
    for i, (x, y, color) in enumerate([(3, -1, "1 .2 .1 1"), (3, 1, ".1 .8 .2 1"), (-3, 0, ".2 .4 1 1"), (0, 3, "1 .8 .1 1")]):
        ET.SubElement(world, "geom", name=f"visual_marker_{i}", type="box", pos=f"{x} {y} .35", size=".15 .15 .35", rgba=color, contype="0", conaffinity="0")
    base = world.find("body[@name='base_link']")
    ET.SubElement(base, "camera", name="front_rgb", pos=".34 0 .05", xyaxes="0 -1 0 .173648 0 .984808", fovy="60")
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))


class Go2Control:
    """Input = normalized forward/turn commands; output = bounded motor torque.

    This hand-written gait uses local kp=80, kd=2, not the competition's
    kp=40 and normalized action interface. No root forces/teleportation used.
    """
    def __init__(self, model):
        self.model = model
        self.qa, self.va = actuator_addresses(model)
        self.actuator_names = [model.actuator(i).name for i in range(model.nu)]
        self.indices = {leg: [self.actuator_names.index(f"{leg}_{j}") for j in ("hip", "thigh", "calf")] for leg in LEGS}
        self.home = model.key_qpos[model.key("home").id, self.qa].copy()
        self.target = self.home.copy()
        self.command = np.zeros(2)
        self.phase_time = 0.0
        self.activity = 0.0

    def update(self, data, forward, turn, dt=.02):
        requested = np.clip([forward, turn], -1, 1)
        self.command += np.clip(requested - self.command, -2 * dt, 2 * dt)
        self.activity += np.clip(float(np.linalg.norm(requested) > .01) - self.activity, -2 * dt, 2 * dt)
        if self.activity < 1e-6:
            self.phase_time = 0.0
            self.target = self.home.copy()
            return self.target
        self.phase_time += dt
        vx, yaw_rate = self.command * [.20, .80]
        period, duty, lift, height = .6, .65, .09, .25
        target = self.home.copy()
        for leg, offset in zip(LEGS, [.5, 0, 0, .5]):
            side = 1 if leg[1] == "L" else -1
            phase = (self.phase_time / period + offset) % 1
            stride = (vx - yaw_rate * side * .142) * period * duty
            if phase < duty:
                x = stride * (.5 - phase / duty)
                z = -height
            else:
                fraction = (phase - duty) / (1 - duty)
                x = stride * (-.5 + fraction)
                z = -height + lift * math.sin(math.pi * fraction)
            knee = -math.acos(float(np.clip((x*x + z*z - 2 * .213**2) / (2 * .213**2), -1, 1)))
            thigh = math.atan2(-x, -z) - knee / 2
            target[self.indices[leg]] = [0, thigh, knee]
        # Smooth transitions between standing and gait trajectories.
        self.target = self.home + self.activity * (target - self.home)
        return self.target

    def apply(self, data):
        torque = 80 * (self.target - data.qpos[self.qa]) - 2 * data.qvel[self.va]
        limited = self.model.actuator_ctrllimited.astype(bool)
        torque[limited] = np.clip(torque[limited], self.model.actuator_ctrlrange[limited, 0], self.model.actuator_ctrlrange[limited, 1])
        data.ctrl[:] = torque


def feedback(model, data, control):
    """Call mj_forward first so sensors and camera refer to current qpos/time."""
    sensors = {model.sensor(i).name: data.sensordata[int(model.sensor_adr[i]):int(model.sensor_adr[i] + model.sensor_dim[i])].tolist() for i in range(model.nsensor)}
    result = {
        "joint_position_rad": data.qpos[control.qa].tolist(),
        "joint_velocity_rad_s": data.qvel[control.va].tolist(),
        "applied_control_torque_nm": data.ctrl.tolist(),
        "actuator_force_nm": data.actuator_force.tolist(),
        "target_joint_position_rad": control.target.tolist(),
        "sensors": sensors,
    }
    # Ground truth is diagnostic only and not an allowed policy observation.
    R = data.xmat[model.body("base_link").id].reshape(3, 3)
    result["debug_ground_truth"] = {
        "base_xyz_m": data.qpos[:3].tolist(),
        "yaw_rad": math.atan2(R[1, 0], R[0, 0]),
        "roll_rad": math.atan2(R[2, 1], R[2, 2]),
        "pitch_rad": math.asin(float(np.clip(-R[2, 0], -1, 1))),
        "contacts": int(data.ncon),
    }
    return result
