#!/usr/bin/env python
import argparse
import bullet_cartpole
import pybullet as p
import random
import numpy as np
import time

# [bbeckman] This is the old code that parses arguments.
# [bbeckman] I'm setting options directly and overwriting command-line options.

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--actions', type=str, default='0,1,2,3,4',
    help='comma seperated list of actions to pick from, if env is discrete')
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

opts.gui = True
discrete_actions = False
env = bullet_cartpole.BulletCartpole(
    opts=opts, discrete_actions=discrete_actions)

EXACT_LQR_CART_POLE_GAINS = [-2.82843,  # x
                             -9.15175,  # x-dot
                             -2.82843,  # y
                             -9.15175,  # y-dot
                             16.0987,   # roll
                             15.3304,   # roll-dot
                             16.0987,   # pitch
                             15.3304]   # pitch-dot

for _ in range(opts.num_eval):
    env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action = random.choice(actions) if discrete_actions \
            else env.action_space.sample()

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

            # debugging code: don't remove too early
            #
            # vos_check0 = env.monkey_velocities[r][0][1]
            # vos_check1 = env.monkey_velocities[r][1][1]
            # vos_check2 = env.monkey_velocities[r][2][1]
            # vos_check3 = env.monkey_velocities[r][3][1]
            # vos_check4 = env.monkey_velocities[r][4][1]
            # vos_mean_check = (vos_check0 + vos_check1 + vos_check2 + \
            #                   vos_check3 + vos_check4) / 5

        steps += 1
        total_reward += reward
        if opts.max_episode_len is not None and steps > opts.max_episode_len:
            break
    print(total_reward)

env.reset()  # hack to flush last event log if required
