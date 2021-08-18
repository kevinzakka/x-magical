import abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np
import pymunk as pm
from gym import spaces

import xmagical.entities as en
import xmagical.render as r
from xmagical.phys_vars import PhysicsVariablesBase, PhysVar
from xmagical.style import ARENA_ZOOM_OUT, COLORS_RGB, lighten_rgb


class PhysicsVariables(PhysicsVariablesBase):
    """Default values & randomisation ranges for key physical parameters of the environment."""

    robot_pos_joint_max_force = PhysVar(5, (3.2, 5.5))
    robot_rot_joint_max_force = PhysVar(1, (0.7, 1.5))
    robot_finger_max_force = PhysVar(4, (2.5, 4.5))
    shape_trans_joint_max_force = PhysVar(1.5, (1.0, 1.8))
    shape_rot_joint_max_force = PhysVar(0.1, (0.07, 0.15))


class BaseEnv(gym.Env, abc.ABC):
    # Constants for all envs.
    ROBOT_RAD = 0.2
    ROBOT_MASS = 1.0
    SHAPE_RAD = ROBOT_RAD * 0.6
    SIZE = 1.1
    ARENA_BOUNDS_LRBT = [-SIZE, SIZE, -SIZE, SIZE]
    ARENA_SIZE_MAX = max(ARENA_BOUNDS_LRBT)
    # Minimum and maximum size of goal regions used during randomisation.
    RAND_GOAL_MIN_SIZE = 0.5
    RAND_GOAL_MAX_SIZE = 0.8
    RAND_GOAL_SIZE_RANGE = RAND_GOAL_MAX_SIZE - RAND_GOAL_MIN_SIZE
    # The following are used to standardise what "jitter" means across different
    # tasks.
    JITTER_PCT = 0.05
    JITTER_POS_BOUND = ARENA_SIZE_MAX * JITTER_PCT / 2.0
    JITTER_ROT_BOUND = JITTER_PCT * np.pi
    JITTER_TARGET_BOUND = JITTER_PCT * RAND_GOAL_SIZE_RANGE / 2

    def __init__(
        self,
        *,  # Subclasses can have additional args.
        robot_cls: Type[en.embodiments.NonHolonomicEmbodiment],
        res_hw: Tuple[int, int] = (256, 256),
        fps: float = 20.0,
        phys_steps: int = 10,
        phys_iter: int = 10,
        max_episode_steps: Optional[int] = None,
        view_mode: str = "allo",
        rand_dynamics: bool = False,
    ) -> None:
        assert view_mode in [
            "allo",
            "ego",
        ], "view_mode must be one of ['allo', 'ego']."

        self.robot_cls = robot_cls
        self.action_dim = robot_cls.DOF
        self.phys_iter = phys_iter
        self.phys_steps = phys_steps
        self.fps = fps
        self.res_hw = res_hw
        self.max_episode_steps = max_episode_steps
        self.rand_dynamics = rand_dynamics

        # State/rendering (see reset()).
        self._entities = None
        self._space = None
        self._robot = None
        self._episode_steps = None
        self._phys_vars = None
        self._renderer_func = (
            self._use_allo_cam if view_mode == "allo" else self._use_ego_cam
        )

        # This is for rendering and displaying.
        self.renderer = None
        self.viewer = None

        # Set observation and action spaces.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(*self.res_hw, 3), dtype=np.uint8
        )
        self.action_space = spaces.Box(
            np.array([-1] * self.action_dim, dtype=np.float32),
            np.array([+1] * self.action_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.seed()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Initialise the PRNG and return seed necessary to reproduce results.

        The action space should probably be seeded in a downstream RL
        application.
        """
        if seed is None:
            seed = np.random.randint(0, (1 << 31) - 1)
        self.rng = np.random.RandomState(seed=seed)
        return [seed]

    def _make_robot(
        self,
        init_pos: Union[np.ndarray, Tuple[float, float]],
        init_angle: float,
    ) -> en.embodiments.NonHolonomicEmbodiment:
        return self.robot_cls(
            radius=self.ROBOT_RAD,
            mass=self.ROBOT_MASS,
            init_pos=init_pos,
            init_angle=init_angle,
        )

    def _make_shape(self, **kwargs) -> en.Shape:
        return en.Shape(shape_size=self.SHAPE_RAD, mass=0.01, **kwargs)

    @abc.abstractmethod
    def on_reset(self) -> None:
        """Set up entities necessary for this environment, and reset any other
        data needed for the env. Must create a robot in addition to any
        necessary entities.
        """
        pass

    def add_entities(self, entities: Sequence[en.Entity]) -> None:
        """Adds a list of entities to the current entities list and sets it up.

        Only intended to be used from within on_reset(). Needs to be called for
        every created entity or else they will not be added to the space!
        """
        for entity in entities:
            if isinstance(entity, self.robot_cls):
                self._robot = entity
            self._entities.append(entity)
            entity.setup(self.renderer, self._space, self._phys_vars)

    def _use_ego_cam(self) -> None:
        """Egocentric agent view."""
        self.renderer.set_cam_follow(
            source_xy_world=(
                self._robot.body.position.x,
                self._robot.body.position.y,
            ),
            target_xy_01=(0.5, 0.15),
            viewport_hw_world=(
                self._arena_h * ARENA_ZOOM_OUT,
                self._arena_w * ARENA_ZOOM_OUT,
            ),
            rotation=self._robot.body.angle,
        )

    def _use_allo_cam(self) -> None:
        """Allocentric 'god-mode' view."""
        self.renderer.set_bounds(
            left=self._arena.left * ARENA_ZOOM_OUT,
            right=self._arena.right * ARENA_ZOOM_OUT,
            bottom=self._arena.bottom * ARENA_ZOOM_OUT,
            top=self._arena.top * ARENA_ZOOM_OUT,
        )

    def reset(self):
        self._episode_steps = 0

        # Delete old entities/space.
        self._entities = []
        self._space = None
        self._robot = None
        self._phys_vars = None

        if self.renderer is None:
            res_h, res_w = self.res_hw
            background_color = lighten_rgb(COLORS_RGB["grey"], times=4)
            self.renderer = r.Viewer(res_w, res_h, background_color)
        else:
            # These will get added back later.
            self.renderer.reset_geoms()

        self._space = pm.Space()
        self._space.collision_slop = 0.01
        self._space.iterations = self.phys_iter

        if self.rand_dynamics:
            # Randomise the physics properties of objects and the robot a
            # little bit.
            self._phys_vars = PhysicsVariables.sample(self.rng)
        else:
            self._phys_vars = PhysicsVariables.defaults()

        # Set up robot and arena.
        arena_l, arena_r, arena_b, arena_t = self.ARENA_BOUNDS_LRBT
        self._arena = en.ArenaBoundaries(
            left=arena_l, right=arena_r, bottom=arena_b, top=arena_t
        )
        self._arena_w = arena_r - arena_l
        self._arena_h = arena_t - arena_b
        self.add_entities([self._arena])

        reset_rv = self.on_reset()
        assert reset_rv is None, (
            f"on_reset method of {type(self)} returned {reset_rv}, but "
            f"should return None"
        )
        assert isinstance(self._robot, self.robot_cls)
        assert len(self._entities) >= 1

        assert np.allclose(self._arena.left + self._arena.right, 0)
        assert np.allclose(self._arena.bottom + self._arena.top, 0)

        self._renderer_func()
        return self.render(mode="rgb_array")

    def _phys_steps_on_frame(self):
        spf = 1 / self.fps
        dt = spf / self.phys_steps
        for i in range(self.phys_steps):
            for ent in self._entities:
                ent.update(dt)
            self._space.step(dt)

    @abc.abstractmethod
    def score_on_end_of_traj(self) -> float:
        """Compute the score for this trajectory.

        Only called at the last step of the trajectory.

        Returns:
           score: number in [0, 1] indicating the worst possible
               performance (0), the best possible performance (1) or something
               in between. Should apply to the WHOLE trajectory.
        """
        pass  # pytype: disable=bad-return-type

    @abc.abstractclassmethod
    def get_reward(self) -> float:
        """Compute the reward for the current timestep.

        This is called at the end of every timestep.
        """
        pass  # pytype: disable=bad-return-type

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._robot.set_action(action)
        self._phys_steps_on_frame()
        self._episode_steps += 1

        obs = self.render(mode="rgb_array")
        reward = self.get_reward()

        done = False
        eval_score = 0.0
        info = {}
        if self.max_episode_steps is not None:
            if self._episode_steps >= self.max_episode_steps:
                info["TimeLimit.truncated"] = not done
                done = True
        if done:
            eval_score = self.score_on_end_of_traj()
            assert (
                0 <= eval_score <= 1
            ), f"eval score {eval_score} out of range for env {self}"
        info.update(eval_score=eval_score)

        return obs, reward, done, info

    def render(self, mode="human") -> Optional[np.ndarray]:
        for ent in self._entities:
            ent.pre_draw()

        self._renderer_func()

        obs = self.renderer.render()
        if mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(obs)
        else:
            return obs

    def close(self) -> None:
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        if self.viewer:
            self.viewer.close()
            self.viewer = None
