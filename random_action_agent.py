#!/usr/bin/env python

# Game-specific

import argparse  # TODO deprecate
import pybullet as p

# General-purpose

import numpy as np
import numpy.random as rnd
import datetime
import time
from functools import partial
from toolz import accumulate
from pprint import PrettyPrinter
from collections import namedtuple as ntup
import os
import json

# Rendering

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

# A/B Learning

from scipy.spatial import distance


class BulletCartpole(object):

    X = 0
    X_DOT = 1
    Y = 2
    Y_DOT = 3
    ROLL = 4
    ROLL_DOT = 5
    PITCH = 6
    PITCH_DOT = 7

    def __init__(self, bullet_cart_id, bullet_pole_id,
                 position_threshold=3.0, angle_threshold=0.35):

        self.cart = bullet_cart_id
        self.pole = bullet_pole_id
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold

        # x, x_dot, y, y_dot, roll, roll_dot, pitch, pitch_dot

        self.state = np.zeros(8)

        # Full information from pybullet, in case we want it.

        self.pso = None
        self.rpy = None
        self.vel = None

    def step(self, action: np.ndarray):

        _info = {}

        p.stepSimulation()

        fx, fy = action
        p.applyExternalForce(
            self.cart, -1, (fx, fy, 0), (0, 0, 0), p.LINK_FRAME)

        self._observe_state()

        _done = False
        if abs(self.state[self.X]) > self.position_threshold \
                or abs(self.state[self.Y]) > self.position_threshold:
            _info['done_reason'] = 'position bounds exceeded'
            _done = True
        elif np.abs(self.state[self.ROLL]) > self.angle_threshold \
                or abs(self.state[self.PITCH]) > self.angle_threshold:
            _info['done_reason'] = 'orientation bounds exceeded'
            _done = True

        _reward = 0.0
        return np.copy(self.state), _reward, _done, _info

    def _observe_state(self):
        self.pso = p.getBasePositionAndOrientation(self.pole)
        self.rpy = p.getEulerFromQuaternion(pso[1])
        self.vel = p.getBaseVelocity(self.pole)
        self.state[self.X] = self.pso[0][0]
        self.state[self.Y] = self.pso[0][1]
        self.state[self.ROLL] = self.rpy[0]
        self.state[self.PITCH] = self.rpy[1]
        self.state[self.X_DOT] = self.vel[0][0]
        self.state[self.Y_DOT] = self.vel[0][1]
        self.state[self.ROLL_DOT] = self.vel[1][0]
        self.state[self.PITCH_DOT] = self.vel[1][1]

    def reset(self):

        # reset pole on cart in starting poses
        p.resetBasePositionAndOrientation(self.cart, (0, 0, 0.08), (0, 0, 0, 1))
        p.resetBasePositionAndOrientation(self.pole, (0, 0, 0.35), (0, 0, 0, 1))

        self._observe_state()
        return np.copy(self.state)


# Processing of Keyboard Commands


def is_a_or_l(c):
    return c == 65 or c == 97 or c == 76 or c == 108


def is_b_or_r(c):
    return c == 66 or c == 98 or c == 82 or c == 114


def is_c(c):
    return c == 67 or c == 99


def is_zero(c):
    return c == 48


def is_six(c):
    return c == 54


def is_plus_or_equals(c):
    return c == ord('+') or c == ord('=')


def is_minus_or_underscore(c):
    return c == ord('-') or c == ord('_')


def is_n(c):
    return c == 78 or c == 110


def is_m(c):
    return c == 77 or c == 109


def is_q(c):
    return c == 81 or c == 113


def is_p(c):
    return c == 80 or c == 112


def is_x(c):
    return c == 88 or c == 120


def is_y(c):
    return c == 89 or c == 121


def is_replay_without_changes(c):
    return command_name(c, None) == 'REPLAY_WITHOUT_CHANGES'


def tighten_search(c):
    return is_a_or_l(c) or is_b_or_r(c) or is_c(c)


def loosen_search(c):
    return is_y(c)


def do_new_search(c):
    return tighten_search(c) or loosen_search(c) or is_n(c)


def command_name(c, ground_truth_mode):
    if is_six(c):
        return ('EXIT_GROUND_TRUTH_MODE'
                if ground_truth_mode else
                'ENTER_GROUND_TRUTH_MODE')
    if is_a_or_l(c):
        return 'PICK_LEFT_AND_TIGHTEN_NEW_SEARCH'
    if is_b_or_r(c):
        return 'PICK_RIGHT_AND_TIGHTEN_NEW_SEARCH'
    if is_c(c):
        return 'PICK_RANDOMLY_AND_TIGHTEN_NEW_SEARCH'
    if is_plus_or_equals(c):
        return 'DOUBLE_DISTURBANCE_AMPLITUDE_AND_REPLAY'
    if is_minus_or_underscore(c):
        return 'HALVE_DISTURBANCE_AMPLITUDE_AND_REPLAY'
    if is_zero(c):
        return 'RESET_DISTURBANCE_AMPLITUDE_AND_REPLAY'
    if is_p(c):
        return 'TOGGLE_REPEATABLE_DISTURBANCE_AND_REPLAY'
    if is_n(c):
        return 'NEW_SEARCH_WITHOUT_TIGHTENING'
    if is_m(c):
        return 'NEW_DISTURBANCE_RUN_WITH_NO_OTHER_CHANGES'
    if is_x(c):
        return 'START_AGAIN_FROM_SCRATCH'
    if is_y(c):
        return 'LOOSEN_NEW_SEARCH'
    if is_q(c):
        return 'QUIT'
    return 'REPLAY_WITHOUT_CHANGES'


def text_to_screen(text, key, screen, rect, fg_color, bg_color):
    b_font = pygame.font.SysFont(None, 48)
    text += ' [ ' + chr(key) + ' ] [' + str(key) + ']'
    b_text = b_font.render(text, True, THECOLORS[fg_color], THECOLORS[bg_color])
    b_rect = b_text.get_rect()
    b_rect.centerx = rect.centerx
    b_rect.centery = rect.centery
    _rect = screen.blit(b_text, b_rect)
    pygame.display.update()


pp = PrettyPrinter(indent=2)
pi2 = np.pi / 2
two_pi = np.pi * 2


def make_pygame_rect(left, top, width, height):
    """Wraps the constructor, adding named arguments."""
    return pygame.Rect(left, top, width, height)


def sum_of_evaluated_funcs(funcs, state, t):
    """I just made a named func for this to inspect in the debugger. A lambda
    is more elegant but less debuggable."""
    results = [f(state, t) for f in funcs]
    result = reduce(operator.add, results)
    return result


def repeatable_disturbance(state, t):  # has form of a 'u' function
    """For comparison with "particularBumpy" in Mathematica."""

    #      5.55208 E ^ (-10(-15 + t) ^ 2)
    r15 = 5.55208 * np.exp(-10. * ((t - 15.) ** 2))

    #    - 4.92702 E ^ (-10(-13 + t) ^ 2)
    r13 = -4.92702 * np.exp(-10. * ((t - 13.) ** 2))

    #    - 4.36616 E ^ (-10(-12 + t) ^ 2)
    r12 = -4.36616 * np.exp(-10. * ((t - 12.) ** 2))

    #    - 4.05894 E ^ (-10(-10 + t) ^ 2)
    r10 = -4.05894 * np.exp(-10. * ((t - 10.) ** 2))

    #    + 1.01364 E ^ (-10(-8 + t) ^ 2)
    r08 = 1.01364 * np.exp(-10. * ((t - 8.) ** 2))

    #    - 3.42841 E ^ (-10(-7 + t) ^ 2)
    r07 = -3.42841 * np.exp(-10. * ((t - 7.) ** 2))

    #    + 4.31046 E ^ (-10(-6 + t) ^ 2)
    r06 = 4.31046 * np.exp(-10. * ((t - 6.) ** 2))

    #    + 3.33288 E ^ (-10(-4 + t) ^ 2)
    r04 = 3.33288 * np.exp(-10. * ((t - 4.) ** 2))

    #    - 5.64323 E ^ (-10(-3 + t) ^ 2)
    r03 = -5.64323 * np.exp(-10. * ((t - 3.) ** 2))

    #    - 7.9857  E ^ (-10(-2 + t) ^ 2)
    r02 = -7.98570 * np.exp(-10. * ((t - 2.) ** 2))

    #    + 1.80146 E ^ (-10 t ^ 2)
    r00 = 1.80146 * np.exp(-10. * ((t - 0.) ** 2))

    return r15 + r13 + r12 + r10 + r08 + r07 + r06 + r04 + r03 + r02 + r00


def very_noisy_disturbance(amplitude=3.0):
    """ Return a state -> time -> force.
    From Mathematica:
    verybumpy[t_] =
    Sum[RandomReal[{-6, 6}]
    Exp[-10(t - RandomInteger[15]) ^ 2], {20}]
    """
    funcs = [lambda state, t: ((2 * amplitude * rnd.rand()) - amplitude)
                              * np.exp((-10 * (t - rnd.randint(0, 16)) ** 2))
             for i in range(20)]
    return partial(sum_of_evaluated_funcs, funcs)


def lqr_control_force(gains, zero_point, state, t_ignore):
    residual = np.array(state) - np.array(zero_point)
    correction = - np.dot(gains, residual)
    return correction


sim_constants_ntup = ntup(
    'SimConstants',
    ['seed', 'dimensions', 'duration_second', 'steps_per_second',
     'initial_condition', 'lqr_zero_point'])

search_constants_ntup = ntup(
    'SearchConstants',
    ['radius', 'decay'])


class GameState(object):
    def __init__(self, pair,
                 output_file_name=None,
                 seed=8420, dimensions=8,
                 duration_second=30, steps_per_second=100,
                 initial_condition=
                 np.array([0, 0, 0, 0, pi2 + 0.1, 0, pi2 + 0.1, 0]),
                 lqr_zero_point=np.array([0, 0, 0, 0, pi2, 0, pi2, 0]),
                 search_covariance_decay=0.975,
                 search_radius=20):
        self.pair = pair
        self.output_file_name = output_file_name

        self.states = None
        self.cov0 = (search_radius ** 2) * np.identity(dimensions)
        self.cov = np.copy(self.cov0)
        self.y0 = np.zeros(dimensions)
        self.yp = np.zeros(dimensions)
        self.ys = [np.zeros(dimensions), np.zeros(dimensions)]
        self.amplitude0 = 1.0
        self.amplitude = 1.0
        self.repeatable_q = True
        self.ground_truth_mode = False
        self.trial_count = 0

        self.max_episode_len = 200

        # threshold for pole position.
        # if absolute x or y moves outside this we finish episode
        self.pos_threshold = 3.0  # TODO: higher?

        # threshold for angle from z-axis.
        # if x or y > this value we finish episode.
        self.angle_threshold = 0.35  # radians; ~= 20deg

        # force to apply per action simulation step.
        # in the discrete case this is the fixed force applied
        # in the continuous case each x/y is in range (-F, F)
        self.action_force = 50.0

        pygame.init()
        self.data_font = pygame.font.SysFont('Consolas', 12)

        self.sim_constants = sim_constants_ntup(
            seed=seed,
            dimensions=dimensions,
            duration_second=duration_second,
            steps_per_second=steps_per_second,
            initial_condition=initial_condition,
            lqr_zero_point=lqr_zero_point)

        # If the seed is zero, it's falsey, and "None" will be passed in,
        # causing non-repeatable pseudo-randoms.

        rnd.seed(self.sim_constants.seed or None)

        self.search_constants = search_constants_ntup(
            decay=search_covariance_decay,
            radius=search_radius)

    def __del__(self):
        pygame.quit()

    def reset(self: 'GameState'):
        self.amplitude = self.amplitude0
        self.cov = np.copy(self.cov0)
        self.yp = np.copy(self.y0)
        self.ys = [self.y0, self.y0]
        self.repeatable_q = True
        self.ground_truth_mode = False

    def manipulate_disturbances(self, c):
        if is_plus_or_equals(c):
            self.amplitude *= 2.0
        elif is_minus_or_underscore(c):
            self.amplitude /= 2.0
        elif is_zero(c):
            self.amplitude = self.amplitude0
        elif is_p(c):
            self.repeatable_q = not self.repeatable_q
        elif is_m(c):
            self.repeatable_q = False

    def process_command_search_mode(self, c):
        if is_a_or_l(c):
            self.yp = np.copy(self.ys[0])
        elif is_b_or_r(c):
            self.yp = np.copy(self.ys[1])
        elif is_c(c):
            self.yp = np.copy(self.ys[rnd.randint(2)])
        else:
            self.manipulate_disturbances(c)
        if do_new_search(c):
            self.new_search()
        if tighten_search(c):
            self.tighten()
        elif loosen_search(c):
            self.loosen()

    def process_command_ground_truth_mode(self, c):
        self.manipulate_disturbances(c)

    def free_cart_pole_lqr(self, i):
        h = 1. / self.sim_constants.steps_per_second

        if self.repeatable_q:
            u = partial(
                sum_of_evaluated_funcs,
                [lambda s, t: repeatable_disturbance(
                    s, t) * self.amplitude / 60,
                 partial(
                     lqr_control_force, self.ys[i],
                     self.sim_constants.lqr_zero_point)])
        else:
            u = partial(
                sum_of_evaluated_funcs,
                [very_noisy_disturbance(self.amplitude),
                 partial(
                     lqr_control_force, self.ys[i],
                     self.sim_constants.lqr_zero_point)])

        n_steps = int(self.sim_constants.duration_second // h)
        time_0 = 0
        solution = list(accumulate(
            partial(step_rk4, partial(self.pair.cart_poles[i].dx_u, u)),
            [(h, h * j) for j in np.arange(n_steps)],
            (time_0, self.sim_constants.initial_condition)
        ))

        return solution

    def solve_pair_motion(self):
        result = [self.free_cart_pole_lqr(i) for i in range(2)]
        return result

    def tighten(self):
        self.cov *= self.search_constants.decay

    def loosen(self):
        self.cov /= self.search_constants.decay

    def new_search(self):
        self.ys[0] = np.array(np.random.multivariate_normal(self.yp, self.cov))
        self.ys[1] = np.array(np.random.multivariate_normal(self.yp, self.cov))

    def record_output(self, c):
        output_dict = \
            {'y_chosen': list(self.yp),  # np.ndarray not json-serializable.
             'time_stamp':
                 f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
             'left_guess': list(self.ys[0]),
             'right_guess': list(self.ys[1]),
             'disturbance_amplitude': self.amplitude,
             'repeatable_disturbance': self.repeatable_q,
             'truth': EXACT_LQR_CART_POLE_GAINS,
             'distance_from_left_guess_to_truth':
                 distance.euclidean(self.ys[0], EXACT_LQR_CART_POLE_GAINS),
             'distance_from_right_guess_to_truth':
                 distance.euclidean(self.ys[1], EXACT_LQR_CART_POLE_GAINS),
             'command': command_name(c, self.ground_truth_mode),
             'dimensions': self.sim_constants.dimensions,
             'sigma': np.sqrt(self.cov[0][0]),
             'trial_count': self.trial_count,
             'output_file_name': self.output_file_name}
        pp.pprint(output_dict)
        jsout = json.dumps(output_dict, indent=2)
        with open(self.output_file_name, "a") as output_file_pointer:
            print(jsout, file=output_file_pointer)
        time.sleep(1)

    def blit_data_text(self):
        self.blit_text(
            f'σ0:            {np.round(np.sqrt(self.cov0[0][0]), 4)}')
        self.blit_text(f'y0:            {np.round(self.y0, 4)}')
        self.blit_text(f'σ:             {np.round(np.sqrt(self.cov[0][0]), 4)}')
        self.blit_text(f'search center: {np.round(self.yp, 4)}')
        self.blit_text(f'left guess     {np.round(self.ys[0], 4)}')
        self.blit_text(f'right guess    {np.round(self.ys[1], 4)}')
        self.blit_text(f'trial number   {self.trial_count}')
        self.blit_text(f'disturbance amplitude {self.amplitude}')
        self.blit_text(f'disturbance is repeatable? {self.repeatable_q}')

    def blit_text(self, the_text):
        b_text = self.data_font.render(
            the_text, True, THECOLORS['white'], THECOLORS['black'])
        b_rect = b_text.get_rect()
        b_rect.left = self.current_left
        b_rect.top = self.current_top
        self.current_top += self.line_spacing
        self.pair.text_surface.blit(b_text, b_rect)

    def render_text(self):
        min_top = 10
        min_left = 10
        self.current_left = min_left
        self.current_top = min_top
        self.line_spacing = 14
        self.column_spacing = (self.pair.screen_width - 20) / 2 - 40

        if self.ground_truth_mode:
            self.blit_text('---- G R O U N D - T R U T H - M O D E -----------')
            self.blit_data_text()
            self.current_left += self.column_spacing
            self.current_top = min_top
            self.blit_text(f'---- C O M M A N D S --------------------- Press:')
            self.blit_text(f"'0' to reset disturbance amplitude and replay")
            self.blit_text(f"'p' to toggle repeatable disturbance and replay")
            self.blit_text(f"")
            self.blit_text(
                f"    ... if both look good and you can't tell which is better, try")
            self.blit_text(
                f"'m' to generate new disturbances, no other changes (implies non-repeatability)")
            self.blit_text(
                f"'+' or '=' to double the disturbance amplitude and replay")
            self.blit_text(
                f"    ... if both look bad and you can't tell which is better, try")
            self.blit_text(
                f"'-' or '_' to halve the disturbance amplitude and replay")
            self.blit_text(f"")
            self.blit_text(
                f"'x' to start over from scratch if you think it's not going to converge")
            self.blit_text(f"'6' to leave ground-truth mode and replay")
            self.blit_text(f"'q' to quit")
            self.blit_text(
                f"or any other key to replay without choosing, tightening, or loosening")
        else:
            self.blit_text('---- S E A R C H - M O D E -----------------------')
            self.blit_data_text()
            self.current_left += self.column_spacing
            self.current_top = min_top
            self.blit_text(f'---- C O M M A N D S --------------------- Press:')
            self.blit_text(
                f"'a' or 'l' to choose left video and tighten new search")
            self.blit_text(
                f"'b' or 'r' to choose right video and tighten new search")
            self.blit_text(f"'c' to choose randomly and tighten new search")
            self.blit_text(
                f"'n' to search again without choosing, tightening, or loosening")
            self.blit_text(f"'y' to loosen new search without choosing")
            self.blit_text(f"'0' to reset disturbance amplitude and replay")
            self.blit_text(f"'p' to toggle repeatable disturbance and replay")
            self.blit_text(f"")
            self.blit_text(
                f"    ... if both look good and you can't tell which is better, try")
            self.blit_text(
                f"'m' to generate new disturbances, no other changes (implies non-repeatability)")
            self.blit_text(
                f"'+' or '=' to double the disturbance amplitude and replay")
            self.blit_text(
                f"    ... if both look bad and you can't tell which is better, try")
            self.blit_text(
                f"'-' or '_' to halve the disturbance amplitude and replay")
            self.blit_text(f"")
            self.blit_text(
                f"'x' to start over from scratch if you think it's not going to converge")
            self.blit_text(f"'6' to enter ground-truth mode and replay")
            self.blit_text(f"'q' to quit")
            self.blit_text(
                f"or any other key to replay without choosing, tightening, or loosening")

        self.pair.screen.blit(self.pair.text_surface, self.pair.text_rect)


def game_factory() -> GameState:

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    p.loadURDF("models/ground.urdf", 0, 0, 0, 0, 0, 0, 1)

    cart1 = p.loadURDF("models/cart.urdf", 0, 0, 0.08, 0, 0, 0, 1)
    pole1 = p.loadURDF("models/pole.urdf", 0, 0, 0.35, 0, 0, 0, 1)

    cart2 = p.loadURDF("models/cart2.urdf", 0, 0, 0.08, 0, 0, 0, 1)
    pole2 = p.loadURDF("models/pole2.urdf", 0, 0, 0.35, 0, 0, 0, 1)

    # [bbeckman] Camera params found by bisective trial-and-error.
    p.resetDebugVisualizerCamera(cameraYaw=0,
                                 cameraTargetPosition=(0, 0, 0),
                                 cameraPitch=-24,
                                 cameraDistance=1.5)

    pair = [BulletCartpole(cart1, pole1), BulletCartpole(cart2, pole2)]

    result = GameState(
        seed=0,
        pair=pair,
        output_file_name=create_place_to_record_results()
    )
    return result


def create_place_to_record_results():
    experiment_path = \
        f"./a_b_lqr_experiments/" + \
        f"{datetime.datetime.now().strftime('day_%Y_%m_%d')}"
    output_file_name = \
        experiment_path + \
        f"""/{datetime.datetime.now().strftime('trial_%H_%M_%S.json')}"""
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    output_file_pointer = open(output_file_name, "w")
    output_file_pointer.close()
    return output_file_name


def keyboard_command_window():
    pygame.init()
    # modes = pygame.display.list_modes()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Keyboard Commands')
    pygame.mouse.set_visible(False)
    bfont = pygame.font.SysFont(None, 48)
    cfont = pygame.font.SysFont('Consolas', 12)

    text_to_screen_center(
        bfont, 'Starting Keyboard Test', None, screen, 'white', 'black')

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == KEYUP:
                text_to_screen_center(
                    bfont, 'KEYUP:', event, screen, 'yellow', 'red')
                print('KEYUP ' + str(event))
                if event.key == pygame.K_q:
                    done = True
            elif event.type == KEYDOWN:
                text_to_screen_center(
                    bfont, 'KEYDOWN:', event, screen, 'blue', 'green')
                print('KEYDOWN ' + str(event))
    pygame.quit()


def text_to_screen_center(bfont, text, event, screen, fgcolor, bgcolor):
    screen.fill(THECOLORS[bgcolor])
    if event is not None:
        text += ' [ ' + chr(event.key) + ' ] [' + str(event.key) + ']'
    btext = bfont.render(text, True, THECOLORS[fgcolor], THECOLORS[bgcolor])
    brect = btext.get_rect()
    brect.centerx = screen.get_rect().centerx
    brect.centery = screen.get_rect().centery
    arect = screen.blit(btext, brect)
    pygame.display.update()


game = game_factory()

EXACT_LQR_CART_POLE_GAINS_X = [-2.82843,  # x
                               -9.15175,  # x-dot
                               0,  # y
                               0,  # y-dot
                               -16.0987,  # roll
                               -15.3304,  # roll-dot
                               0,  # pitch
                               0]  # pitch-dot

EXACT_LQR_CART_POLE_GAINS_Y = [0,  # x
                               0,  # x-dot
                               -2.82843,  # y
                               -9.15175,  # y-dot
                               0,  # roll
                               0,  # roll-dot
                               -16.0987,  # pitch
                               -15.3304]  # pitch-dot

quit()

while True:
    solutions = game.solve_pair_motion()
    steps = zip(solutions[0], solutions[1])
    [game.render([step[0][1], step[1][1]]) for step in steps]
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == KEYUP:
                c = event.key
                done = True

    text_to_screen(
        command_name(c, game.ground_truth_mode), c,
        game.pair.screen, game.pair.text_rect, 'cyan', 'blue')

    if is_q(c):
        break
    elif is_six(c):
        if not game.ground_truth_mode:
            ys_saved = copy.deepcopy(game.ys)
            game.ys = copy.deepcopy(
                [EXACT_LQR_CART_POLE_GAINS, EXACT_LQR_CART_POLE_GAINS])
            game.ground_truth_mode = True
        else:
            game.ys = copy.deepcopy(ys_saved)
            game.ground_truth_mode = False
    elif is_x(c):
        game.reset()
        continue

    if game.ground_truth_mode:
        game.process_command_ground_truth_mode(c)
    else:
        game.trial_count += 1
        game.process_command_search_mode(c)
        game.record_output(c)


# [bbeckman] This is the old code that parses arguments.

opts.repeats = 1
opts.gui = True
opts.delay = 0.0125 / 2
opts.action_force = 1.0
discrete_actions = False
env = bullet_cartpole.BulletCartpole(
    opts=opts, discrete_actions=discrete_actions)
# [bbeckman] These aren't exact, just shots in the dark, but not completely
# nuts.
for _ in range(opts.num_eval):
    env.reset()
    done = False
    total_reward = 0
    steps = 0
    physical_pole_states_set_point = np.array(
        [[0, 0, 0, 0, -np.pi / 2, 0, np.pi / 2, 0],
         [0, 0, 0, 0, -np.pi / 2, 0, np.pi / 2, 0]])
    physical_pole_states = np.array(
        [[0, 0, 0, 0, -np.pi / 2 + 0.1, 0, np.pi / 2 + 0.1, 0],
         [0, 0, 0, 0, -np.pi / 2 + 0.1, 0, np.pi / 2 + 0.1, 0]])
    x_actions = np.array([None, None])
    y_actions = np.array([None, None])
    actions = np.array([None, None])
    while not done:
        residuals = physical_pole_states_set_point - physical_pole_states
        for i in range(2):
            x_actions[i] = np.dot(residuals[i], EXACT_LQR_CART_POLE_GAINS_X)
            y_actions[i] = np.dot(residuals[i], EXACT_LQR_CART_POLE_GAINS_Y)
            actions[i] = np.array([x_actions[i], y_actions[i]])

        # TODO: [bbeckman] make this the physical state. Repulse the monkeys.
        _state, reward, done, info = env.step(actions)

        # [bbeckman] Get physical state from this "gym" state.
        # q = _state[0][1][-4:]
        # normalcy_check = np.dot(q, q)
        # assert np.isclose(normalcy_check, 1.0)

        # TODO: [bbeckman] get rid of 'repeats'
        # TODO: [bbeckman] understand why there are five 'steps_per_repeat'
        for r in range(env.repeats):
            for i in range(2):
                pos_means = np.mean(env.monkey_positions[i][r], axis=0)
                mean_pos = pos_means[0]
                mean_vel = pos_means[1]
                ang_means = np.mean(env.monkey_velocities[i][r], axis=0)
                mean_rpy = ang_means[0]
                mean_rpy_dot = ang_means[1]

                physical_pole_states[i] = np.array(
                    [mean_pos[0], mean_vel[0],
                     mean_pos[1], mean_vel[1],
                     mean_rpy[0], mean_rpy_dot[0],
                     mean_rpy[1], mean_rpy_dot[1],
                     ])

        steps += 1
        total_reward += reward
        if opts.max_episode_len is not None and steps > opts.max_episode_len:
            break
    print(total_reward)
    keyboard_command_window()

env.reset()  # hack to flush last event log if required
