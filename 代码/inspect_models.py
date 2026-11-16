"""Report model topology and frozen solver settings without changing assets."""
import json
import mujoco
from check_resources import RESOURCES, ROOT, verify


def describe(model):
    return {
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "nbody": model.nbody, "ngeom": model.ngeom, "ncam": model.ncam,
        "nkey": model.nkey, "timestep": model.opt.timestep,
        "solver": int(model.opt.solver), "cone": int(model.opt.cone),
        "iterations": model.opt.iterations, "ls_iterations": model.opt.ls_iterations,
        "impratio": model.opt.impratio,
        "joints": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)],
        "actuators": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)],
        "cameras": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)],
        "keys": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)],
    }


if __name__ == "__main__":
    report = {"resources": verify(), "mujoco": mujoco.__version__}
    for name, path in {"go2": "robot/go2/physics/go2.xml", "course": "scene/mujoco/competition-course.mjb"}.items():
        p = RESOURCES / path
        model = mujoco.MjModel.from_binary_path(str(p)) if p.suffix == ".mjb" else mujoco.MjModel.from_xml_path(str(p))
        report[name] = describe(model)
    out = ROOT / "输出/仿真"
    out.mkdir(parents=True, exist_ok=True)
    (out / "models.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
