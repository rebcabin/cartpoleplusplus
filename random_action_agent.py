#!/usr/bin/env python
import argparse
import bullet_cartpole
import pybullet as p
import random
import numpy as np
import numpy.random as rnd
import copy
import time

# [bbeckman] This is the old code that parses arguments.
# [bbeckman] I'm setting options directly and overwriting command-line options.

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--actions', type=str, default='0,1,2,3,4',
    help='comma-separated list of actions to pick from, if env is discrete')
parser.add_argument(
    '--num-eval', type=int, default=1000)
parser.add_argument(
    '--action-type', type=str, default='discrete',
    help="either 'discrete' or 'continuous'")
bullet_cartpole.add_opts(parser)
opts = parser.parse_args()

actions = list(map(int, opts.actions.split(",")))

if opts.action_type == 'discrete':
    discrete_actions = True
elif opts.action_type == 'continuous':
    discrete_actions = False
else:
    raise Exception(f"Unknown action type [{opts.action_type}]")

opts.repeats = 1
opts.gui = True
opts.delay = 0.0125
opts.action_force = 1.0
discrete_actions = False
env = bullet_cartpole.BulletCartpole(
    opts=opts, discrete_actions=discrete_actions)
# [bbeckman] These aren't exact, just shots in the dark, but not completely
# nuts.
EXACT_LQR_CART_POLE_GAINS_X = [-2.82843,  # x
                               -9.15175,  # x-dot
                                      0,  # y
                                      0,  # y-dot
                               -16.0987,  # roll
                               -15.3304,  # roll-dot
                                      0,  # pitch
                                      0]  # pitch-dot

EXACT_LQR_CART_POLE_GAINS_Y = [       0,  # x
                                      0,  # x-dot
                               -2.82843,  # y
                               -9.15175,  # y-dot
                                      0,  # roll
                                      0,  # roll-dot
                               -16.0987,  # pitch
                               -15.3304]  # pitch-dot

for _ in range(opts.num_eval):
    env.reset()
    done = False
    total_reward = 0
    steps = 0
    physical_state_set_point = np.array([0, 0, 0, 0, -np.pi/2, 0, np.pi/2, 0])
    physical_state = np.array([0, 0, 0, 0, -np.pi/2 + 0.1, 0, np.pi/2 + 0.1, 0])
    while not done:
        action = random.choice(actions) if discrete_actions \
            else env.action_space.sample()

        residual = physical_state_set_point - physical_state
        x_action = np.dot(residual, EXACT_LQR_CART_POLE_GAINS_X)
        y_action = np.dot(residual, EXACT_LQR_CART_POLE_GAINS_Y)

        action = np.array([[x_action, y_action]])

        action = np.array([[0, 0]])

        _state, reward, done, info = env.step(action)

        # [bbeckman] Get physical state from this "gym" state.
        q = _state[0][1][-4:]
        normalcy_check = np.dot(q, q)
        assert np.isclose(normalcy_check, 1.0)

        for r in range(env.repeats):
            pos_means = np.mean(env.monkey_positions[r], axis=0)
            mean_pos = pos_means[0]
            mean_vel = pos_means[1]
            ang_means = np.mean(env.monkey_velocities[r], axis=0)
            mean_rpy = ang_means[0]
            mean_rpy_dot = ang_means[1]

            physical_state = [mean_pos[0], mean_vel[0],
                              mean_pos[1], mean_vel[1],
                              mean_rpy[0], mean_rpy_dot[0],
                              mean_rpy[1], mean_rpy_dot[1],
                              ]

        steps += 1
        total_reward += reward
        if opts.max_episode_len is not None and steps > opts.max_episode_len:
            break
    print(total_reward)

env.reset()  # hack to flush last event log if required
