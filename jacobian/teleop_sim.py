import json
import socket
import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "robot.xml"
EE_SITE_NAME = "ee"

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
SOCKET_TIMEOUT = 0.001

CTRL_RATE = 100.0
DAMPING = 1e-2
JOINT_VEL_LIMIT = 1.5
COMMAND_TIMEOUT = 0.25  # seconds

INITIAL_QPOS = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)


def damped_pseudoinverse_qdot(jacp: np.ndarray, v_cmd: np.ndarray, damping: float) -> np.ndarray:
    """
    qdot = J^T (J J^T + lambda^2 I)^(-1) v
    """
    jj_t = jacp @ jacp.T
    reg = (damping ** 2) * np.eye(3)
    return jacp.T @ np.linalg.solve(jj_t + reg, v_cmd)


# UDP receive socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(SOCKET_TIMEOUT)

print(f"Listening for controller packets on {UDP_IP}:{UDP_PORT}")

# MuJoCo init
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
if site_id == -1:
    raise ValueError(f"Site '{EE_SITE_NAME}' not found in model")

print("Model nq:", model.nq, "nv:", model.nv, "nu:", model.nu)
print("EE site id:", site_id)

if model.nq < 4 or model.nv < 4:
    raise ValueError("This script expects a 4-DOF fixed-base arm.")

data.qpos[:4] = INITIAL_QPOS
mujoco.mj_forward(model, data)

latest_cmd = np.zeros(3, dtype=float)
last_cmd_time = 0.0
quit_requested = False

dt = 1.0 / CTRL_RATE
last_print = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and not quit_requested:
        loop_start = time.time()

        # Receive newest command if available
        try:
            while True:
                packet, _addr = sock.recvfrom(4096)
                msg = json.loads(packet.decode("utf-8"))
                latest_cmd = np.array([msg["vx"], msg["vy"], msg["vz"]], dtype=float)
                last_cmd_time = time.time()
                quit_requested = bool(msg.get("quit", False))
        except socket.timeout:
            pass
        except BlockingIOError:
            pass

        # Zero command if controller stream stops
        if time.time() - last_cmd_time > COMMAND_TIMEOUT:
            latest_cmd[:] = 0.0

        # Kinematics and Jacobian
        mujoco.mj_forward(model, data)
        x_ee = data.site_xpos[site_id].copy()

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        qdot = damped_pseudoinverse_qdot(jacp, latest_cmd, DAMPING)
        qdot = np.clip(qdot, -JOINT_VEL_LIMIT, JOINT_VEL_LIMIT)

        # Simple first version: directly integrate qpos
        data.qpos[:4] += qdot[:4] * dt
        mujoco.mj_forward(model, data)

        now = time.time()
        if now - last_print > 0.2:
            print(
                "cmd:", np.round(latest_cmd, 3),
                "| qdot:", np.round(qdot[:4], 3),
                "| ee:", np.round(x_ee, 3)
            )
            last_print = now

        viewer.sync()

        elapsed = time.time() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

sock.close()
print("MuJoCo teleop exiting.")