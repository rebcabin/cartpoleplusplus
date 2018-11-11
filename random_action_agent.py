#!/usr/bin/env python
import argparse
import bullet_cartpole
import pybullet as p
import random
import numpy as np
import time

# [bbeckman] I'm setting options directly and overwriting command-line options.

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--actions', type=str, default='0,1,2,3,4',
                    help='comma seperated list of actions to pick from, if env is discrete')
parser.add_argument('--num-eval', type=int, default=1000)
parser.add_argument('--action-type', type=str, default='discrete',
                    help="either 'discrete' or 'continuous'")
bullet_cartpole.add_opts(parser)
opts = parser.parse_args()

actions = list(map(int, opts.actions.split(",")))

if opts.action_type == 'discrete':
    discrete_actions = True
elif opts.action_type == 'continuous':
    discrete_actions = False
else:
    raise Exception("Unknown action type [%s]" % opts.action_type)

opts.gui = True
discrete_actions = False
env = bullet_cartpole.BulletCartpole(opts=opts,
                                     discrete_actions=discrete_actions)

for _ in range(opts.num_eval):
    env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        if discrete_actions:
            action = random.choice(actions)
        else:
            action = env.action_space.sample()
        _state, reward, done, info = env.step(action)
        # [bbeckman] Get physical state from this "gym" state.
        q = _state[0][1][-4:]
        normalcy_check = np.dot(q, q)
        assert np.isclose(normalcy_check, 1.0)
        steps += 1
        total_reward += reward
        if opts.max_episode_len is not None and steps > opts.max_episode_len:
            break
    print(total_reward)

env.reset()  # hack to flush last event log if required
