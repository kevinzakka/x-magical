"""A user interface for teleoperating an agent in an x-magical environment.

Modified from https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/recorder/env_interactor.py
"""

import time
from typing import List

import numpy as np
import pyglet.window
from gym.envs.classic_control.rendering import SimpleImageViewer
from PIL import Image

UP_DOWN_MAG = 0.5
ANGLE_MAG = np.radians(1.5)
OPEN_CLOSE_MAG = np.pi / 8


class KeyboardEnvInteractor(SimpleImageViewer):
    """User interface for interacting in an x-magical environment."""

    def __init__(
        self,
        action_dim: int,
        fps: float = 20,
        resolution: int = 384,
    ):
        super().__init__(maxwidth=resolution)

        self._action_dim = action_dim
        self._dt = 1.0 / fps
        self._resolution = resolution
        self._keys = pyglet.window.key.KeyStateHandler()

        self.reset()

    def imshow(self, image: np.ndarray) -> None:
        self._last_image = image
        was_none = self.window is None
        image = Image.fromarray(image)
        image = image.resize((self._resolution, self._resolution))
        image = np.array(image)
        super().imshow(image)
        if was_none:
            self.window.event(self.on_key_press)
            self.window.push_handlers(self._keys)

    def get_action(self) -> List[float]:
        action = [0.0, 0.0, 0.0]

        if self._keys[pyglet.window.key.UP] and not self._keys[pyglet.window.key.DOWN]:
            action[0] = +UP_DOWN_MAG
            self._started = True
        elif (
            self._keys[pyglet.window.key.DOWN] and not self._keys[pyglet.window.key.UP]
        ):
            action[0] = -UP_DOWN_MAG
            self._started = True
        if (
            self._keys[pyglet.window.key.LEFT]
            and not self._keys[pyglet.window.key.RIGHT]
        ):
            action[1] = ANGLE_MAG
            self._started = True
        elif (
            self._keys[pyglet.window.key.RIGHT]
            and not self._keys[pyglet.window.key.LEFT]
        ):
            action[1] = -ANGLE_MAG
            self._started = True
        if self._keys[pyglet.window.key.SPACE]:
            action[2] = OPEN_CLOSE_MAG
            self._started = True

        if self._keys[pyglet.window.key.ESCAPE]:
            self._finish_early = True

        return action[: self._action_dim]

    def on_key_press(self, x, y):
        return True

    def reset(self):
        self._started = False
        self._finish_early = False
        self._last_image = None

    def run_loop(self, step_fn):
        """Run an environment interaction loop.

        The step_fn will be continually called with actions, and it should
        return observations. When step_fn returns None, the loop is done.
        """
        last_time = time.time()
        while not self._finish_early:
            action = self.get_action()
            if self._started:
                obs = step_fn(action)
                if obs is None:
                    return
                self.imshow(obs)
            else:
                # Needed to run the event loop.
                self.imshow(self._last_image)
            pyglet.clock.tick()  # pytype: disable=module-attr
            delta = time.time() - last_time
            time.sleep(max(0, self._dt - delta))
            last_time = time.time()
