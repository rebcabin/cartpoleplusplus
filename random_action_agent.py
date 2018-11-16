# Game-specific

import pybullet as p

# General-purpose

import numpy as np
import numpy.random as rnd
import datetime
import time
from functools import partial
from toolz import reduce
from pprint import PrettyPrinter
from collections import namedtuple as ntup
import os
import json

# Windowing

import os
import tkinter as tk

# Rendering

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

# A/B Learning

from scipy.spatial import distance


class BulletCartpole(object):

    """It might seem natural to have the thresholds maintained outside this
    class, however, it mimics the Gym interface of OpenAI. Its 'step' method
    is responsible for reporting when it's done. Therefore, 'step' must know
    these thresholds."""

    X = 0
    X_DOT = 1
    Y = 2
    Y_DOT = 3
    ROLL = 4
    ROLL_DOT = 5
    PITCH = 6
    PITCH_DOT = 7

    def __init__(self, bullet_cart_id, bullet_pole_id,
                 position_threshold, angle_threshold,
                 initial_pole_state, lqr_zero_point,
                 initial_cart_bullet_state, initial_pole_bullet_state):

        self.cart = bullet_cart_id
        self.pole = bullet_pole_id

        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold

        # x, x_dot, y, y_dot, roll, roll_dot, pitch, pitch_dot

        self.initial_pole_state = np.copy(initial_pole_state)
        self.pole_state = np.copy(initial_pole_state)
        self.lqr_zero_point = np.copy(lqr_zero_point)

        self.initial_cart_bullet_state = initial_cart_bullet_state
        self.initial_pole_bullet_state = initial_pole_bullet_state

        # Full information from pybullet, in case we want it.

        self.pso = None
        self.rpy = None
        self.vel = None

        # Verified that, at least at initialization time,
        # the observed (physics) state matches the bullet state.

        self._observe_state()

    def lqr_control_forces(self, controller):
        residual = self.pole_state - self.lqr_zero_point
        corrections = (- np.dot(controller, residual))
        return corrections

    def step(self, action: np.ndarray):
        p.stepSimulation()
        fx, fy = action
        p.applyExternalForce(
            self.cart, -1, (fx, fy, 0), (0, 0, 0), p.LINK_FRAME)
        self._observe_state()
        _done, _info = self._check_done()
        _reward = 0.0
        return np.copy(self.pole_state), _reward, _done, _info

    def _check_done(self):
        _done = False
        _info = {}
        if abs(self.pole_state[self.X]) > self.position_threshold \
                or abs(self.pole_state[self.Y]) > self.position_threshold:
            _done = True
            _info['done_reason'] = 'position bounds exceeded'
        elif np.abs(self.pole_state[self.ROLL]) > self.angle_threshold \
                or abs(self.pole_state[self.PITCH]) > self.angle_threshold:
            _done = True
            _info['done_reason'] = 'orientation bounds exceeded'
        return _done, _info

    def _observe_state(self):
        self.pso = p.getBasePositionAndOrientation(self.pole)
        self.rpy = p.getEulerFromQuaternion(self.pso[1])
        self.vel = p.getBaseVelocity(self.pole)
        self.pole_state[self.X] = self.pso[0][0]
        self.pole_state[self.Y] = self.pso[0][1]
        self.pole_state[self.ROLL] = self.rpy[0]
        self.pole_state[self.PITCH] = self.rpy[1]
        self.pole_state[self.X_DOT] = self.vel[0][0]
        self.pole_state[self.Y_DOT] = self.vel[0][1]
        self.pole_state[self.ROLL_DOT] = self.vel[1][0]
        self.pole_state[self.PITCH_DOT] = self.vel[1][1]

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.cart,
            self.initial_cart_bullet_state[0:3],
            self.initial_cart_bullet_state[3:])
        p.resetBasePositionAndOrientation(
            self.pole,
            self.initial_pole_bullet_state[0:3],
            self.initial_pole_bullet_state[3:])
        self._observe_state()
        return np.copy(self.pole_state)


# Processing of Keyboard Commands


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


def repeatable_disturbance(_state, t):  # has form of a 'u' function
    """For comparison with "particularBumpy" in Mathematica."""

    #      5.55208 E ^ (-10(-15 + t) ^ 2)
    r15 =  5.55208 * np.exp(-10. * ((t - 15.) ** 2))

    #    - 4.92702 E ^ (-10(-13 + t) ^ 2)
    r13 = -4.92702 * np.exp(-10. * ((t - 13.) ** 2))

    #    - 4.36616 E ^ (-10(-12 + t) ^ 2)
    r12 = -4.36616 * np.exp(-10. * ((t - 12.) ** 2))

    #    - 4.05894 E ^ (-10(-10 + t) ^ 2)
    r10 = -4.05894 * np.exp(-10. * ((t - 10.) ** 2))

    #    + 1.01364 E ^ (-10(-8 + t) ^ 2)
    r08 =  1.01364 * np.exp(-10. * ((t - 8.) ** 2))

    #    - 3.42841 E ^ (-10(-7 + t) ^ 2)
    r07 = -3.42841 * np.exp(-10. * ((t - 7.) ** 2))

    #    + 4.31046 E ^ (-10(-6 + t) ^ 2)
    r06 =  4.31046 * np.exp(-10. * ((t - 6.) ** 2))

    #    + 3.33288 E ^ (-10(-4 + t) ^ 2)
    r04 =  3.33288 * np.exp(-10. * ((t - 4.) ** 2))

    #    - 5.64323 E ^ (-10(-3 + t) ^ 2)
    r03 = -5.64323 * np.exp(-10. * ((t - 3.) ** 2))

    #    - 7.9857  E ^ (-10(-2 + t) ^ 2)
    r02 = -7.98570 * np.exp(-10. * ((t - 2.) ** 2))

    #    + 1.80146 E ^ (-10 t ^ 2)
    r00 =  1.80146 * np.exp(-10. * ((t - 0.) ** 2))

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


sim_constants_ntup = ntup(
    'SimConstants',
    ['seed',
     'state_dimensions', 'action_dimensions',
     'duration_second', 'steps_per_second', 'delta_time',
     'action_force_multiplier'])

search_constants_ntup = ntup(
    'SearchConstants',
    ['radius', 'decay'])

command_screen_constants_ntup = ntup(
    'CommandScreenConstants',
    ['width', 'height']
)


class GameState(object):

    def __init__(
            self, pair,
            output_file_name=None,
            seed=8420,

            state_dimensions=8,
            action_dimensions=2,

            duration_second=30,
            steps_per_second=60,
            delta_time=1.0 / 60.0,  # TODO: pybullet default time ?

            action_force_multiplier=5.0,

            search_covariance_decay=0.975,
            search_radius=20,

            command_screen_width=600,
            command_screen_height=600):
        self.pair = pair
        self.output_file_name = output_file_name

        self.cov0 = (search_radius ** 2) * np.identity(state_dimensions)
        self.cov = np.copy(self.cov0)
        self.y0 = np.zeros((action_dimensions, state_dimensions))
        self.yp = np.zeros((action_dimensions, state_dimensions))
        self.ys = [np.copy(self.y0), np.copy(self.y0)]

        self.amplitude0 = 1.0
        self.amplitude = 1.0

        self.repeatable_q = True

        self.ground_truth_mode = False

        self.trial_count = 0

        self.sim_constants = sim_constants_ntup(
            seed=seed,
            state_dimensions=state_dimensions,
            action_dimensions=action_dimensions,
            duration_second=duration_second,
            steps_per_second=steps_per_second,
            delta_time=delta_time,
            action_force_multiplier=action_force_multiplier)

        # If the seed is zero, it's falsey, and "None" will be passed in,
        # causing non-repeatable pseudo-randoms.

        rnd.seed(self.sim_constants.seed or None)

        self.search_constants = search_constants_ntup(
            decay=search_covariance_decay,
            radius=search_radius)

        self.cmdwin = command_screen_constants_ntup(
            height=command_screen_height,
            width=command_screen_width)

        # Empty slots for pygame control of the command window.

        self.screen = None
        self.text_surface = None
        self.text_rect = None
        self.dpy_font = None
        self.data_font = None

        # More window-control stuff

        self.pygame_inited = False
        self.tk_root = None

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
        if GameState.is_plus_or_equals(c):
            self.amplitude *= 2.0
        elif GameState.is_minus_or_underscore(c):
            self.amplitude /= 2.0
        elif GameState.is_zero(c):
            self.amplitude = self.amplitude0
        elif GameState.is_p(c):
            self.repeatable_q = not self.repeatable_q
        elif GameState.is_m(c):
            self.repeatable_q = False

    def command_name(self, c):
        if GameState.is_six(c):
            return ('EXIT_GROUND_TRUTH_MODE'
                    if self.ground_truth_mode else
                    'ENTER_GROUND_TRUTH_MODE')
        if GameState.is_a_or_l(c):
            return 'PICK_LEFT_AND_TIGHTEN_NEW_SEARCH'
        if GameState.is_b_or_r(c):
            return 'PICK_RIGHT_AND_TIGHTEN_NEW_SEARCH'
        if GameState.is_c(c):
            return 'PICK_RANDOMLY_AND_TIGHTEN_NEW_SEARCH'
        if GameState.is_plus_or_equals(c):
            return 'DOUBLE_DISTURBANCE_AMPLITUDE_AND_REPLAY'
        if GameState.is_minus_or_underscore(c):
            return 'HALVE_DISTURBANCE_AMPLITUDE_AND_REPLAY'
        if GameState.is_zero(c):
            return 'RESET_DISTURBANCE_AMPLITUDE_AND_REPLAY'
        if GameState.is_p(c):
            return 'TOGGLE_REPEATABLE_DISTURBANCE_AND_REPLAY'
        if GameState.is_n(c):
            return 'NEW_SEARCH_WITHOUT_TIGHTENING'
        if GameState.is_m(c):
            return 'NEW_DISTURBANCE_RUN_WITH_NO_OTHER_CHANGES'
        if GameState.is_x(c):
            return 'START_AGAIN_FROM_SCRATCH'
        if GameState.is_y(c):
            return 'LOOSEN_NEW_SEARCH'
        if GameState.is_q(c):
            return 'QUIT'
        return 'REPLAY_WITHOUT_CHANGES'

    def process_command_search_mode(self, c):
        if GameState.is_a_or_l(c):
            self.yp = np.copy(self.ys[0])
        elif GameState.is_b_or_r(c):
            self.yp = np.copy(self.ys[1])
        elif GameState.is_c(c):
            self.yp = np.copy(self.ys[rnd.randint(2)])
        else:
            self.manipulate_disturbances(c)
        if GameState.do_new_search(c):
            self.new_search()
        if GameState.tighten_search(c):
            self.tighten()
        elif GameState.loosen_search(c):
            self.loosen()
        _result = [p.reset() for p in self.pair]

    def process_command_ground_truth_mode(self, c):
        self.manipulate_disturbances(c)
        _result = [p.reset() for p in self.pair]

    def tighten(self):
        self.cov *= self.search_constants.decay

    def loosen(self):
        self.cov /= self.search_constants.decay

    def new_search(self):
        self.ys[0][0] = \
            np.array(np.random.multivariate_normal(self.yp[0], self.cov))
        self.ys[0][1] = \
            np.array(np.random.multivariate_normal(self.yp[1], self.cov))

        self.ys[1][0] = \
            np.array(np.random.multivariate_normal(self.yp[0], self.cov))
        self.ys[1][1] = \
            np.array(np.random.multivariate_normal(self.yp[1], self.cov))

    def record_output(self, c):
        # np.ndarray not json-serializable.
        output_dict = \
            {'y_chosen': list(map(list, self.yp)),
             'time_stamp':
                 f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
             'left_guess': list(map(list, self.ys[0])),
             'right_guess': list(map(list, self.ys[1])),
             'disturbance_amplitude': self.amplitude,
             'repeatable_disturbance': self.repeatable_q,
             'truth': [EXACT_GAINS_X, EXACT_GAINS_Y],
             'distance_from_left_guess_to_truth':
                 [distance.euclidean(self.ys[0][0], EXACT_GAINS_X),
                  distance.euclidean(self.ys[0][1], EXACT_GAINS_Y)],
             'distance_from_right_guess_to_truth':
                 [distance.euclidean(self.ys[1][0], EXACT_GAINS_X),
                  distance.euclidean(self.ys[1][1], EXACT_GAINS_Y)],
             'command': self.command_name(c),
             'state_dimensions': self.sim_constants.state_dimensions,
             'action_dimensions': self.sim_constants.action_dimensions,
             'sigma': np.sqrt(self.cov[0][0]),
             'trial_count': self.trial_count,
             'output_file_name': self.output_file_name}
        pp.pprint(output_dict)
        js_out = json.dumps(output_dict, indent=2)
        with open(self.output_file_name, "a") as output_file_pointer:
            print(js_out, file=output_file_pointer)
        time.sleep(1)

    def render_data(self):
        sigma0 = np.round(np.sqrt(self.cov0[0][0]), 4)
        sigma = np.round(np.sqrt(self.cov[0][0]), 4)
        self.blit_line(f'σ0:            {sigma0}')
        self.blit_line(f'y0:            {np.round(self.y0, 4)}')
        self.blit_line(f'σ:             {sigma}')
        self.blit_line(f'search center: {np.round(self.yp, 4)}')
        self.blit_line(f'left guess     {np.round(self.ys[0], 4)}')
        self.blit_line(f'right guess    {np.round(self.ys[1], 4)}')
        self.blit_line(f'trial number   {self.trial_count}')
        self.blit_line(f'disturbance amplitude {self.amplitude}')
        self.blit_line(f'disturbance is repeatable? {self.repeatable_q}')

    def blit_line(self, the_text):
        b_text = self.data_font.render(
            the_text, True, THECOLORS['white'], THECOLORS['black'])
        b_rect = b_text.get_rect()
        b_rect.left = self.current_left
        b_rect.top = self.current_top
        self.current_top += self.line_spacing
        self.text_surface.blit(b_text, b_rect)

    def render_text(self):
        min_top = 10
        min_left = 10
        self.current_left = min_left
        self.current_top = min_top
        self.line_spacing = 14
        self.column_spacing = (self.cmdwin.width - 20) / 2 - 40

        if self.ground_truth_mode:
            self.blit_line('---- G R O U N D - T R U T H - M O D E -----------')
            self.render_data()
            # self.current_left += self.column_spacing
            # self.current_top = min_top
            self.blit_line(f"")
            self.blit_line(f'---- C O M M A N D S --------------------- Press:')
            self.blit_line(f"'0' to reset disturbance amplitude and replay")
            self.blit_line(f"'p' to toggle repeatable disturbance and replay")
            self.blit_line(f"")
            self.blit_line(
                f"    ... if both look good and you can't tell which is better, try")
            self.blit_line(
                f"'m' to generate new disturbances, no other changes (implies non-repeatability)")
            self.blit_line(
                f"'+' or '=' to double the disturbance amplitude and replay")
            self.blit_line(
                f"    ... if both look bad and you can't tell which is better, try")
            self.blit_line(
                f"'-' or '_' to halve the disturbance amplitude and replay")
            self.blit_line(f"")
            self.blit_line(
                f"'x' to start over from scratch if you think it's not going to converge")
            self.blit_line(f"'6' to leave ground-truth mode and replay")
            self.blit_line(f"'q' to quit")
            self.blit_line(
                f"or any other key to replay without choosing, tightening, or loosening")
        else:
            self.blit_line('---- S E A R C H - M O D E -----------------------')
            self.render_data()
            # self.current_left += self.column_spacing
            # self.current_top = min_top
            self.blit_line(f"")
            self.blit_line(f'---- C O M M A N D S --------------------- Press:')
            self.blit_line(
                f"'a' or 'l' to choose left video and tighten new search")
            self.blit_line(
                f"'b' or 'r' to choose right video and tighten new search")
            self.blit_line(f"'c' to choose randomly and tighten new search")
            self.blit_line(
                f"'n' to search again without choosing, tightening, or loosening")
            self.blit_line(f"'y' to loosen new search without choosing")
            self.blit_line(f"'0' to reset disturbance amplitude and replay")
            self.blit_line(f"'p' to toggle repeatable disturbance and replay")
            self.blit_line(f"")
            self.blit_line(
                f"    ... if both look good and you can't tell which is better, try")
            self.blit_line(
                f"'m' to generate new disturbances, no other changes (implies non-repeatability)")
            self.blit_line(
                f"'+' or '=' to double the disturbance amplitude and replay")
            self.blit_line(
                f"    ... if both look bad and you can't tell which is better, try")
            self.blit_line(
                f"'-' or '_' to halve the disturbance amplitude and replay")
            self.blit_line(f"")
            self.blit_line(
                f"'x' to start over from scratch if you think it's not going to converge")
            self.blit_line(f"'6' to enter ground-truth mode and replay")
            self.blit_line(f"'q' to quit")
            self.blit_line(
                f"or any other key to replay without choosing, tightening, or loosening")

        self.screen.blit(self.text_surface, self.text_rect)

    def keyboard_command_window(self):
        if not self.pygame_inited:
            self.tk_root = tk.Tk()
            embed = tk.Frame(
                self.tk_root,
                width=self.cmdwin.width, height=self.cmdwin.height)
            embed.pack()
            os.environ['SDL_WINDOWID'] = str(embed.winfo_id())
            os.environ['SDL_VIDEODRIVER'] = 'windib'  # needed on windows.
            self.tk_root.geometry("+0+0")
            self.tk_root.update()

            pygame.init()
            # modes = pygame.display.list_modes()
            self.screen = pygame.display.set_mode(self.cmdwin)  # as tuple
            pygame.display.set_caption('Keyboard Commands')
            pygame.mouse.set_visible(False)
            self.dpy_font = pygame.font.SysFont(None, 48)
            self.data_font = pygame.font.SysFont('Consolas', 12)
            self.text_surface = pygame.Surface(self.cmdwin)
            self.text_surface.fill(THECOLORS['black'])
            self.text_rect = pygame.Rect(
                0, 0, self.cmdwin.width, self.cmdwin.height)
            self.pygame_inited = True

        self.render_text()
        pygame.display.flip()

        done = False
        result = None
        while not done:
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == K_SPACE:
                    rw = rnd.randint(0, 800)
                    rh = rnd.randint(0, 600)
                    str2 = f"+{rw}+{rh}"

                if event.type == KEYUP:
                    self.command_blast(event.key)
                    done = True
                    result = event.key

        return result

    def command_blast(self, key, fg_color='green', bg_color='blue'):
        text = \
            self.command_name(key) + ' [ ' + chr(key) + ' ] [' + str(key) + ']'
        b_text = self.dpy_font.render(
            #    |True| = antialiased
            text, True, THECOLORS[fg_color], THECOLORS[bg_color])
        b_rect = b_text.get_rect()
        b_rect.centerx = self.text_rect.centerx
        b_rect.centery = self.text_rect.centery
        _rect = self.screen.blit(b_text, b_rect)
        pygame.display.update()

    @staticmethod
    def is_a_or_l(c):
        return c == 65 or c == 97 or c == 76 or c == 108

    @staticmethod
    def is_b_or_r(c):
        return c == 66 or c == 98 or c == 82 or c == 114

    @staticmethod
    def is_c(c):
        return c == 67 or c == 99

    @staticmethod
    def is_zero(c):
        return c == 48

    @staticmethod
    def is_six(c):
        return c == 54

    @staticmethod
    def is_plus_or_equals(c):
        return c == ord('+') or c == ord('=')

    @staticmethod
    def is_minus_or_underscore(c):
        return c == ord('-') or c == ord('_')

    @staticmethod
    def is_n(c):
        return c == 78 or c == 110

    @staticmethod
    def is_m(c):
        return c == 77 or c == 109

    @staticmethod
    def is_q(c):
        return c == 81 or c == 113

    @staticmethod
    def is_p(c):
        return c == 80 or c == 112

    @staticmethod
    def is_x(c):
        return c == 88 or c == 120

    @staticmethod
    def is_y(c):
        return c == 89 or c == 121

    def is_replay_without_changes(self, c):
        return self.command_name(c, None) == 'REPLAY_WITHOUT_CHANGES'

    @staticmethod
    def tighten_search(c):
        return GameState.is_a_or_l(c) \
               or GameState.is_b_or_r(c) \
               or GameState.is_c(c)

    @staticmethod
    def loosen_search(c):
        return GameState.is_y(c)

    @staticmethod
    def do_new_search(c):
        return GameState.tighten_search(c) \
               or GameState.loosen_search(c) \
               or GameState.is_n(c)


def game_factory() -> GameState:

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    p.loadURDF("models/ground.urdf", 0, 0, 0, 0, 0, 0, 1)

    initial_cart1_bullet_state = -0.5, 0, 0.08, 0, 0, 0, 1
    cart1 = p.loadURDF("models/cart.urdf",  *initial_cart1_bullet_state)

    initial_pole1_bullet_state = -0.5, 0, 0.35, 0, 0, 0, 1
    pole1 = p.loadURDF("models/pole.urdf",  *initial_pole1_bullet_state)

    initial_cart2_bullet_state =  0.5, 0, 0.08, 0, 0, 0, 1
    cart2 = p.loadURDF("models/cart2.urdf", *initial_cart2_bullet_state)

    initial_pole2_bullet_state =  0.5, 0, 0.35, 0, 0, 0, 1
    pole2 = p.loadURDF("models/pole2.urdf", *initial_pole2_bullet_state)

    # [bbeckman] Camera params found by bisective trial-and-error.
    p.resetDebugVisualizerCamera(cameraYaw=0,
                                 cameraTargetPosition=(0, 0, 0),
                                 cameraPitch=-24,
                                 cameraDistance=1.5)

    position_threshold = 3.0
    angle_threshold = 0.35

    pair = [
        BulletCartpole(
            cart1, pole1,
            position_threshold, angle_threshold,
            initial_pole_state=
            #           X   X' Y  Y' roll  roll'  pitch  pitch'
            np.array([-0.5, 0, 0, 0, 0,    0,     0,     0]),
            lqr_zero_point=
            np.array([-0.5, 0, 0, 0, 0,    0,     0,     0]),
            initial_cart_bullet_state=initial_cart1_bullet_state,
            initial_pole_bullet_state=initial_pole1_bullet_state
        ),
        BulletCartpole(
            cart2, pole2,
            position_threshold, angle_threshold,
            initial_pole_state=
            np.array([+0.5, 0, 0, 0, 0,    0,     0,     0]),
            lqr_zero_point=
            np.array([+0.5, 0, 0, 0, 0,    0,     0,     0]),
            initial_cart_bullet_state=initial_cart2_bullet_state,
            initial_pole_bullet_state=initial_pole2_bullet_state
        )]

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


EXACT_GAINS_X = [-2.82843,  # x
                 -9.15175,  # x-dot
                 0,  # y
                 0,  # y-dot
                 -16.0987,  # roll
                 -15.3304,  # roll-dot
                 0,  # pitch
                 0]  # pitch-dot


EXACT_GAINS_Y = [0,  # x
                 0,  # x-dot
                 -2.82843,  # y
                 -9.15175,  # y-dot
                 0,  # roll
                 0,  # roll-dot
                 -16.0987,  # pitch
                 -15.3304]  # pitch-dot


game = game_factory()


theta = pi2 / 2
sin_theta = np.sin(theta)
cos_theta = np.cos(theta)


while True:
    for step in range(game.sim_constants.duration_second *
                      game.sim_constants.steps_per_second):
        t = step * game.sim_constants.delta_time
        action_scalar = repeatable_disturbance(None, t) \
            * game.sim_constants.action_force_multiplier
        fx = cos_theta * action_scalar
        fy = sin_theta * action_scalar
        force = np.array([fx, fy])
        control = [game.pair[i].lqr_control_forces(game.ys[i])
                   for i in range(2)]
        _state0, _rwd0, _done0, _info0 = game.pair[0].step(force + control[0])
        _state1, _rwd1, _done1, _info1 = game.pair[1].step(force + control[1])
        time.sleep(game.sim_constants.delta_time / 2)
        if _done0 and _done1:
            break
    c = game.keyboard_command_window()

    if GameState.is_q(c):
        break
    elif GameState.is_six(c):
        if not game.ground_truth_mode:
            ys_saved = np.copy(game.ys)
            game.ys = np.copy([
                [EXACT_GAINS_X, EXACT_GAINS_Y],
                [EXACT_GAINS_X, EXACT_GAINS_Y]])
            game.ground_truth_mode = True
        else:
            # If the game logic is wrong, the reference to ys_saved
            # will intentionally blow up.
            game.ys = np.copy(ys_saved)
            game.ground_truth_mode = False
    elif GameState.is_x(c):
        game.reset()
        continue

    if game.ground_truth_mode:
        game.process_command_ground_truth_mode(c)
    else:
        game.trial_count += 1
        game.process_command_search_mode(c)
        game.record_output(c)

pygame.quit()

