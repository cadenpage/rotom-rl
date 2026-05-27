onshape_api:
    ./.venv/bin/onshape-to-robot .

test_mujoco:
    ./.venv/bin/mjpython ./.venv/bin/onshape-to-robot-mujoco .

train:
    ./.venv/bin/python basline.py train --timesteps 50000

eval:
    ./.venv/bin/python basline.py eval --episodes 10

play:
    ./.venv/bin/mjpython basline.py play --xml-path scene.xml --episodes 3

plot:
    ./.venv/bin/python basline.py plot-rollout --episodes 3 --output-path artifacts/rotom_rollout.png

controller:
    ./.venv/bin/python jacobian/controller.py

teleop:
    ./.venv/bin/mjpython jacobian/teleop_sim.py
