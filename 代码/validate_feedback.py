"""Check recorded sensor/image alignment and actual command response."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    records = [json.loads(line) for line in (path / "feedback.jsonl").read_text().splitlines()]
    frames = list(csv.DictReader((path / "frames.csv").open()))
    assert summary["status"] == "CONTROL_FEEDBACK_OK", summary
    assert len(records) == summary["feedback_records"]
    assert len(frames) == summary["camera_frames"]
    for index, record in enumerate(records):
        assert record["feedback_index"] == index
        assert record["physics_step"] == index * 10
        assert record["sim_time_ns"] == index * 20_000_000
        assert record["sim_time_s"] == record["sim_time_ns"] / 1e9
        assert set(record["sensors"]) == {s["name"] for s in manifest["sensor_schema"]}
        for s in manifest["sensor_schema"]:
            values = record["sensors"][s["name"]]
            assert len(values) == s["dimension"] and np.isfinite(values).all()
        for a, name in enumerate(manifest["actuator_order"]):
            assert np.isclose(record["sensors"][name + "_pos"][0], record["joint_position_rad"][a], atol=1e-10)
            assert np.isclose(record["sensors"][name + "_vel"][0], record["joint_velocity_rad_s"][a], atol=1e-10)
        assert np.isclose(np.linalg.norm(record["sensors"]["imu_quat"]), 1, atol=1e-8)
        assert np.max(np.abs(record["applied_control_torque_nm"])) <= 45.43 + 1e-8
    image_hashes = []
    for i, frame in enumerate(frames):
        assert int(frame["frame_index"]) == i
        assert int(frame["sim_time_ns"]) == i * 100_000_000
        record = records[int(frame["feedback_index"])]
        assert int(frame["sim_time_ns"]) == record["sim_time_ns"]
        assert int(frame["physics_step"]) == record["physics_step"]
        assert frame["relative_path"] == record["camera_frame"]["relative_path"]
        image = path / frame["relative_path"]
        assert hashlib.sha256(image.read_bytes()).hexdigest() == frame["sha256"] == record["camera_frame"]["sha256"]
        pixels = np.asarray(Image.open(image))
        assert pixels.shape == (240, 320, 3) and pixels.dtype == np.uint8
        assert pixels.std() > 2, "Blank camera image"
        image_hashes.append(frame["sha256"])
    if manifest["camera"]["enabled"]:
        assert len(frames) == (len(records) - 1) // 5 + 1
        assert len(set(image_hashes)) > 1
    initial_yaw = records[0]["debug_ground_truth"]["yaw_rad"]
    delta = summary["delta_xyz_m"]
    delta_x = np.cos(initial_yaw) * delta[0] + np.sin(initial_yaw) * delta[1]
    yaw = summary["delta_yaw_rad"]
    mode = summary["mode"]
    if mode == "forward":
        assert delta_x > .15, f"Forward motion insufficient: {delta_x}"
    elif mode == "backward":
        assert delta_x < -.15, f"Backward motion insufficient: {delta_x}"
    elif mode == "left":
        assert yaw > .3, f"Left turn insufficient: {yaw}"
    elif mode == "right":
        assert yaw < -.3, f"Right turn insufficient: {yaw}"
    elif mode == "probe":
        motor = manifest["actuator_order"].index("FR_thigh")
        before = np.mean([r["joint_position_rad"][motor] for r in records if .6 <= r["sim_time_s"] <= .9])
        during = np.mean([r["joint_position_rad"][motor] for r in records if 1.4 <= r["sim_time_s"] <= 1.9])
        assert during - before > .015, f"Joint feedback did not respond: {during-before}"
        summary["probe_joint_response_rad"] = float(during - before)
    elif mode == "demo":
        def truth_at(t):
            return records[round(t / .02)]["debug_ground_truth"]
        def forward_delta(t1, t2):
            a, b = np.array(truth_at(t1)["base_xyz_m"]), np.array(truth_at(t2)["base_xyz_m"])
            return float((b - a)[:2] @ [np.cos(initial_yaw), np.sin(initial_yaw)])
        assert forward_delta(1, 6) > .15
        assert forward_delta(7, 12) < -.15
        yaws = np.unwrap([r["debug_ground_truth"]["yaw_rad"] for r in records])
        assert yaws[850] - yaws[650] > .3
        assert yaws[1100] - yaws[900] < -.3
    return {"session": path.name, "validation": "PASS", "sensor_values_per_record": sum(s["dimension"] for s in manifest["sensor_schema"]),
            "timestamp_alignment": "exact", "distinct_camera_frames": len(set(image_hashes)), **summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = [validate(Path(path)) for path in args.sessions]
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)
