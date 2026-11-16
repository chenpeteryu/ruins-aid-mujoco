"""Keyboard/scripted control with synchronized RGB and sensor recording."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

import mujoco
import numpy as np
from PIL import Image

from basic_sim import reset
from check_resources import ROOT, verify
from robot_control import Go2Control, feedback, make_environment, update_front_camera

MODES = {"forward": (1, 0), "backward": (-1, 0), "left": (0, 1), "right": (0, -1), "stand": (0, 0), "probe": (0, 0)}


def scripted(mode, seconds):
    if mode == "demo":
        for end, command, label in [(1, (0, 0), "stand"), (6, (1, 0), "forward"),
                (7, (0, 0), "stop"), (12, (-1, 0), "backward"),
                (13, (0, 0), "stop"), (17, (0, 1), "left"),
                (18, (0, 0), "stop"), (22, (0, -1), "right")]:
            if seconds < end:
                return command, label
        return (0, 0), "stop"
    if seconds < 1:
        return (0, 0), "settle"
    return MODES[mode], mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[*MODES, "demo", "keyboard"], default="demo")
    parser.add_argument("--scene", choices=["flat", "course"], default="flat")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    args = parser.parse_args()
    seconds = args.seconds if args.seconds is not None else (24 if args.mode == "demo" else 60 if args.mode == "keyboard" else 10)
    if not math.isfinite(seconds) or seconds < .1:
        parser.error("--seconds must be finite and >= 0.1")
    if args.mode == "keyboard":
        args.viewer = True
    resources = verify()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "输出/操控反馈" / f"{stamp}-{args.scene}-{args.mode}-{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True)
    (output / "camera").mkdir()
    model = make_environment(args.scene)
    data = reset(model)
    control = Go2Control(model)
    mutex = threading.Lock()
    state = {"command": (0, 0), "label": "stop", "quit": False}
    def key_callback(key):
        with mutex:
            keys = {87: ((1, 0), "forward"), 83: ((-1, 0), "backward"), 65: ((0, 1), "left"), 68: ((0, -1), "right"), 32: ((0, 0), "stop")}
            if key in keys:
                state["command"], state["label"] = keys[key]
            if key == 81:
                state["quit"] = True
    sensor_schema = [{"name": model.sensor(i).name, "dimension": int(model.sensor_dim[i]), "type": int(model.sensor_type[i])} for i in range(model.nsensor)]
    manifest = {
        "schema": "go2-local-feedback/1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "requested_seconds": seconds, "mujoco": mujoco.__version__,
        "resources": resources, "scene": args.scene,
        "local_spawn_xyz": data.qpos[:3].tolist(),
        "official_start_position": False,
        "clock": "sim_time_ns = physics_step * 2000000; all feedback and RGB from same post-step mj_forward state",
        "physics_dt_s": .002, "control_dt_s": .02, "feedback_hz": 50,
        "camera": {"enabled": not args.no_camera, "name": "front_rgb", "width": 320, "height": 240,
                   "hz": 10, "fovy_deg": 60, "base_position_m": [.34, 0, .05], "pitch_down_deg": 10,
                   "format": "RGB uint8 PNG", "official_sensor_contract": False,
                   "render_backend": os.environ.get("MUJOCO_GL", "glfw")},
        "controller": {"type": "hand_written_joint_PD_diagonal_gait", "kp": 80, "kd": 2,
                       "period_s": .6, "stance_fraction": .65, "lift_m": .09,
                       "command_semantics": "normalized direction, not calibrated velocity setpoint",
                       "official_action_interface": False, "root_force_or_teleportation": False},
        "actuator_order": control.actuator_names, "sensor_schema": sensor_schema,
        "debug_ground_truth": "diagnostics only; not used by gait controller",
        "keyboard": "W forward, S backward, A left, D right, Space stop, Q finish; commands latch until next key",
    }
    manifest["camera"]["implementation"] = "body_attached_virtual_GL_camera" if args.scene == "course" else "model_camera"
    manifest["camera"]["terrain_collision_group_3_visible"] = args.scene == "course"
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"SESSION={output}", flush=True)
    if args.mode == "keyboard":
        print(manifest["keyboard"], flush=True)
    renderer = None
    viewer = None
    records = []
    frames = 0
    success = False
    start_wall = time.monotonic()
    last_cmd = (0, 0)
    try:
        if not args.no_camera:
            renderer = mujoco.Renderer(model, height=240, width=320)
        if args.viewer:
            import mujoco.viewer as mjviewer
            viewer = mjviewer.launch_passive(model, data, key_callback=key_callback)
            if args.scene == "course":
                viewer.opt.geomgroup[3] = 1
            viewer.cam.distance = 2.8
            viewer.cam.elevation = -20
        with (output / "feedback.jsonl").open("w", buffering=1) as log, (output / "frames.csv").open("w", newline="") as framefile:
            writer = csv.writer(framefile, lineterminator="\n")
            writer.writerow(["frame_index", "physics_step", "sim_time_ns", "feedback_index", "relative_path", "sha256"])
            count = round(seconds / .02)
            for tick in range(count + 1):
                physics_step = tick * 10
                sim_ns = physics_step * 2_000_000
                if abs(data.time - sim_ns / 1e9) > 1e-8:
                    raise RuntimeError("Simulation clock mismatch (possible instability reset)")
                mujoco.mj_forward(model, data)
                sample = feedback(model, data, control)
                sample.update(feedback_index=tick, physics_step=physics_step, sim_time_ns=sim_ns,
                              sim_time_s=sim_ns / 1e9, requested_command_previous_step=list(last_cmd),
                              smoothed_command=control.command.tolist(),
                              wall_elapsed_ns=time.monotonic_ns() - int(start_wall * 1e9), camera_frame=None)
                if renderer is not None and tick % 5 == 0:
                    relative = f"camera/{frames:06d}_{sim_ns:012d}.png"
                    update_front_camera(renderer, model, data, args.scene)
                    pixels = renderer.render()
                    if pixels.shape != (240, 320, 3):
                        raise RuntimeError("Unexpected camera shape")
                    Image.fromarray(pixels).save(output / relative)
                    digest = hashlib.sha256((output / relative).read_bytes()).hexdigest()
                    sample["camera_frame"] = {"index": frames, "relative_path": relative, "sim_time_ns": sim_ns, "sha256": digest}
                    writer.writerow([frames, physics_step, sim_ns, tick, relative, digest])
                    frames += 1
                log.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                records.append(sample)
                truth = sample["debug_ground_truth"]
                if truth["base_xyz_m"][2] < .14 or max(abs(truth["roll_rad"]), abs(truth["pitch_rad"])) > .8:
                    raise RuntimeError("Robot fell; stopping and preserving synchronized data")
                if any(w.number for w in data.warning):
                    raise RuntimeError("MuJoCo physics warning")
                if tick == count:
                    break
                if args.mode == "keyboard":
                    with mutex:
                        command, label, quit_requested = state["command"], state["label"], state["quit"]
                    if quit_requested:
                        break
                else:
                    command, label = scripted(args.mode, tick * .02)
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    with viewer.lock():
                        viewer.cam.lookat[:] = data.xpos[model.body("base_link").id]
                    viewer.sync()
                if tick % 25 == 0:
                    print(f"t={tick*.02:5.2f}s {label:8s} xyz={np.round(data.qpos[:3],3)} gyro={np.round(data.sensor('imu_gyro').data,3)} frames={frames}", flush=True)
                control.update(data, *command)
                if args.mode == "probe" and 1 <= tick * .02 < 2:
                    control.target[control.indices["FR"][1]] += .06
                for _ in range(10):
                    control.apply(data)
                    mujoco.mj_step(model, data)
                last_cmd = command
                if viewer is not None:
                    time.sleep(max(0, start_wall + (tick + 1) * .02 - time.monotonic()))
        success = True
    finally:
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()
        if records:
            yaws = np.unwrap([r["debug_ground_truth"]["yaw_rad"] for r in records])
            summary = {"status": "CONTROL_FEEDBACK_OK" if success else "FAILED", "mode": args.mode, "scene": args.scene,
                       "sim_seconds": records[-1]["sim_time_s"], "feedback_records": len(records), "camera_frames": frames,
                       "delta_xyz_m": (np.array(records[-1]["debug_ground_truth"]["base_xyz_m"]) - records[0]["debug_ground_truth"]["base_xyz_m"]).tolist(),
                       "delta_yaw_rad": float(yaws[-1] - yaws[0]),
                       "min_base_height_m": min(r["debug_ground_truth"]["base_xyz_m"][2] for r in records),
                       "max_abs_roll_pitch_rad": max(max(abs(r["debug_ground_truth"]["roll_rad"]), abs(r["debug_ground_truth"]["pitch_rad"])) for r in records)}
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
