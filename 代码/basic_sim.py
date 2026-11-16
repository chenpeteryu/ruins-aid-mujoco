"""Reproducible Go2 standing smoke test; this is not a locomotion policy."""
import argparse
import csv
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from check_resources import RESOURCES, ROOT, verify


def load_model(scene):
    if mujoco.__version__ != "3.17.0":
        raise RuntimeError(f"Expected MuJoCo 3.17.0, got {mujoco.__version__}")
    if scene == "course":
        source = RESOURCES / "scene/mujoco/competition-course.mjb"
        model = mujoco.MjModel.from_binary_path(str(source))
    else:
        source = RESOURCES / "robot/go2/physics/go2.xml"
        xml = ET.parse(source).getroot()
        xml.find("compiler").set("meshdir", str(source.parent / "assets"))
        world = xml.find("worldbody")
        ET.SubElement(world, "geom", name="local_test_floor", type="plane", size="5 5 0.1", rgba="0.55 0.6 0.65 1")
        ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", diffuse="0.8 0.8 0.8")
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    # Apply the participant guide's baseline only to the in-memory local model.
    model.opt.timestep = 0.002
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 50
    model.opt.ls_iterations = 10
    model.opt.impratio = 100
    if model.nu != 12:
        raise RuntimeError("Expected 12 actuators")
    if scene == "course":
        # The distributed keyframe sits at the boundary corner. This is a local
        # smoke-test spawn, not an official start coordinate or task contract.
        home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        model.key_qpos[home, :3] = [1.0, 1.0, 0.3]
    return model, source


def reset(model):
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home < 0:
        raise RuntimeError("Missing home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home)
    # Keyframe ctrl contains joint angles, whereas these actuators accept torque.
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    # Small foot/ground penetrations are normal; wall/body overlaps are not.
    for contact in data.contact[:data.ncon]:
        if contact.dist < -0.03:
            raise RuntimeError(f"Initial penetration is too large: {contact.dist}")
    return data


def actuator_addresses(model):
    if np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
        raise RuntimeError("Only direct joint actuators are supported")
    joints = model.actuator_trnid[:, 0]
    if np.any(model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE):
        raise RuntimeError("Expected hinge joints")
    if not np.allclose(model.actuator_gear[:, 0], 1):
        raise RuntimeError("Expected unit actuator gear")
    return model.jnt_qposadr[joints], model.jnt_dofadr[joints]


def run(model, seconds, callback=None):
    data = reset(model)
    initial = data.qpos.copy()
    qa, va = actuator_addresses(model)
    target = initial[qa].copy()
    trajectory = []
    controls = round(seconds / 0.02)
    for step in range(controls):
        for _ in range(10):
            torque = 40 * (target - data.qpos[qa]) - 2 * data.qvel[va]
            limited = model.actuator_ctrllimited.astype(bool)
            torque[limited] = np.clip(torque[limited], model.actuator_ctrlrange[limited, 0], model.actuator_ctrlrange[limited, 1])
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Nonfinite physics state")
        trajectory.append([float(data.time), *data.qpos[:3].tolist(), int(data.ncon)])
        if callback:
            callback(model, data, step, controls)
    warnings = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
    if warnings:
        raise RuntimeError(f"MuJoCo warnings: {warnings}")
    displacement = float(np.linalg.norm(data.qpos[:2] - initial[:2]))
    height = float(data.qpos[2])
    if not 0.15 < height < 0.5 or displacement > 0.2:
        raise RuntimeError(f"Standing failed: height={height}, drift={displacement}")
    if not np.isclose(data.time, controls * 0.02):
        raise RuntimeError("Simulation clock mismatch")
    return data, trajectory, {
        "sim_seconds": float(data.time), "control_steps": controls,
        "physics_steps": controls * 10, "final_xyz": data.qpos[:3].tolist(),
        "xy_drift_m": displacement, "warning_counts": warnings,
        "state_sha256": hashlib.sha256(data.qpos.tobytes() + data.qvel.tobytes()).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=["flat", "course"], default="flat")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--render", action="store_true")
    p.add_argument("--viewer", action="store_true")
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0.02:
        p.error("--seconds must be finite and >= 0.02")
    resources = verify()
    model, source = load_model(args.scene)
    out = ROOT / "输出/仿真"
    out.mkdir(parents=True, exist_ok=True)
    renderer = None
    viewer = None
    images = []
    visual_options = mujoco.MjvOption()
    if args.scene == "course":
        # The public MJB stores terrain in collision group 3, hidden by default.
        visual_options.geomgroup[3] = 1
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    def camera(data, overview=False):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = model.stat.center if overview else data.xpos[base_id]
        cam.distance = 18.0 if overview else 2.2
        cam.azimuth = 135
        cam.elevation = -45 if overview else -20
        return cam
    try:
        if args.render:
            from PIL import Image
            renderer = mujoco.Renderer(model, height=480, width=640)
            d = reset(model)
            renderer.update_scene(d, camera=camera(d, overview=args.scene == "course"), scene_option=visual_options)
            Image.fromarray(renderer.render()).save(out / f"{args.scene}-overview.png")
        if args.viewer:
            import mujoco.viewer as mjviewer
            viewer_data = reset(model)
            viewer = mjviewer.launch_passive(model, viewer_data)
            viewer.opt.geomgroup[:] = visual_options.geomgroup
            viewer.cam.lookat[:] = viewer_data.xpos[base_id]
            viewer.cam.distance = 2.5
        def frame(m, d, step, total):
            if renderer is not None and step % 5 == 0:
                renderer.update_scene(d, camera=camera(d), scene_option=visual_options)
                from PIL import Image
                image = Image.fromarray(renderer.render())
                if len(images) < 100:
                    images.append(image)
                image.save(out / f"{args.scene}-standing.png")
            if viewer is not None and viewer.is_running():
                viewer_data.qpos[:] = d.qpos
                viewer_data.qvel[:] = d.qvel
                mujoco.mj_forward(m, viewer_data)
                viewer.sync()
                time.sleep(0.02)
        start = time.monotonic()
        _, trajectory, report = run(model, args.seconds, frame)
        report["wall_seconds_including_render"] = time.monotonic() - start
        _, second_trajectory, second = run(model, args.seconds)
        if report["state_sha256"] != second["state_sha256"] or trajectory != second_trajectory:
            raise RuntimeError("Reset/replay determinism check failed")
        if renderer is not None:
            pixels = np.asarray(images[-1])
            if pixels.std() < 2:
                raise RuntimeError("Rendered image appears blank")
            if images:
                images[0].save(out / f"{args.scene}-standing.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
        report.update({"status": "BASIC_SIM_OK", "scene": args.scene,
                       "mujoco": mujoco.__version__, "resources": resources,
                       "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                       "deterministic_reset_replay": True, "render_verified": args.render,
                       "render_backend": os.environ.get("MUJOCO_GL", "glfw") if args.render else None,
                       "controller": "home-position PD kp=40 kd=2; zero policy action; no locomotion",
                       "actuator_joint_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in model.actuator_trnid[:, 0]],
                       "control_period": 0.02, "timestep": model.opt.timestep})
        report["local_spawn_xyz"] = reset(model).qpos[:3].tolist()
        with (out / f"{args.scene}-trajectory.csv").open("w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["sim_time", "base_x", "base_y", "base_z", "contacts"])
            writer.writerows(trajectory)
        (out / f"{args.scene}-result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
