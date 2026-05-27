import json
import socket
import time
import pygame

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
SEND_RATE_HZ = 50
DEADBAND = 0.08
LINEAR_SPEED = 0.20  # m/s command scale

xbox_axes = {
    "left_x": 0,
    "left_y": 1,
    "right_x": 2,
    "right_y": 3,
    "lt": 4,
    "rt": 5,
}

xbox_buttons = {
    "a": 0,
    "b": 1,
    "x": 2,
    "y": 3,
    "spc_l": 4,
    "xbox": 5,
    "spc_r": 6,
    "L": 7,
    "R": 8,
    "lb": 9,
    "rb": 10,
    "up": 11,
    "down": 12,
    "left": 13,
    "right": 14,
    "dpad_left": 15,
}


def apply_deadband(value: float, deadband: float) -> float:
    return 0.0 if abs(value) < deadband else value


def trigger_to_unit_interval(raw_value: float) -> float:
    # common mapping: [-1, 1] -> [0, 1]
    return 0.5 * (raw_value + 1.0)


pygame.init()
pygame.joystick.init()

count = pygame.joystick.get_count()
print("joysticks found:", count)

if count == 0:
    print("No controller connected.")
    pygame.quit()
    raise SystemExit

js = pygame.joystick.Joystick(0)
js.init()

print("name:", js.get_name())
print("num axes:", js.get_numaxes())
print("num buttons:", js.get_numbuttons())

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

running = True
clock = pygame.time.Clock()

try:
    while running:
        pygame.event.pump()

        left_x = apply_deadband(js.get_axis(xbox_axes["left_x"]), DEADBAND)
        left_y = apply_deadband(js.get_axis(xbox_axes["left_y"]), DEADBAND)
        lt_raw = js.get_axis(xbox_axes["lt"])
        rt_raw = js.get_axis(xbox_axes["rt"])

        lt = trigger_to_unit_interval(lt_raw)
        rt = trigger_to_unit_interval(rt_raw)

        vx = LINEAR_SPEED * left_x
        vy = LINEAR_SPEED * (-left_y)
        vz = LINEAR_SPEED * (rt - lt)

        buttons = {name: js.get_button(idx) for name, idx in xbox_buttons.items()}

        msg = {
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "quit": bool(buttons["b"]),
            "time": time.time(),
        }

        sock.sendto(json.dumps(msg).encode("utf-8"), (UDP_IP, UDP_PORT))

        print(
            f"vx={vx:+.3f} vy={vy:+.3f} vz={vz:+.3f} "
            f"| quit={msg['quit']}",
            end="\r",
            flush=True,
        )

        if buttons["b"]:
            running = False

        clock.tick(SEND_RATE_HZ)

finally:
    print("\nController exiting.")
    pygame.quit()
    sock.close()