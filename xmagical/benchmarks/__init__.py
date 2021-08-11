import collections
import dataclasses
import enum
import itertools
from typing import Callable, Dict, Tuple

import gym

from xmagical.entities import embodiments

__all__ = [
    "ALL_REGISTERED_ENVS",
    "DEMO_ENVS_TO_TEST_ENVS_MAP",
    "register_envs",
]


# ======================================= #
# Enum definitions.
# ======================================= #


class Task(str, enum.Enum):
    """All tasks currently supported."""

    SWEEP_TO_TOP = "SweepToTop"


class Embodiment(str, enum.Enum):
    """All embodiments currently supported."""

    GRIPPER = "Gripper"
    SHORTSTICK = "Shortstick"
    MEDIUMSTICK = "Mediumstick"
    LONGSTICK = "Longstick"


class ObservationSpace(str, enum.Enum):
    """The type of observation used by a task environment."""

    PIXELS = "Pixels"
    STATE = "State"


class ViewMode(str, enum.Enum):
    """The viewing mode used by the agent."""

    ALLOCENTRIC = "Allo"
    EGOCENTRIC = "Ego"


class Variant(str, enum.Enum):
    """Aspects of the task that can be randomized."""

    # No randomization.
    DEMO = "Demo"
    # Rotations and orientations of all objects, and the size of goal regions,
    # is jittered by up to 5% of the maximum range.
    JITTER = "TestJitter"
    # Positions and rotations of all objects are randomized.
    LAYOUT = "TestLayout"
    # Colors of blocks and goal regions are randomized.
    COLOR = "TestColor"
    # Shapes of pushable blocks are randomized.
    SHAPE = "TestShape"
    # Mass and friction of objects is randomized.
    DYNAMICS = "TestDynamics"
    # All applicable randomizations are applied.
    ALL = "TestAll"


# ======================================= #
# Helper maps.
# ======================================= #

TASK_TO_EPOINT: Dict[Task, str] = {
    Task.SWEEP_TO_TOP: "xmagical.benchmarks.sweep_to_top:SweepToTopEnv",
}
# Whether the task supports state-based observations.
TASK_TO_STATE_AVAILABILITY: Dict[Task, bool] = {
    Task.SWEEP_TO_TOP: True,
}
# The variants the task supports. Not all tasks support all variants.
TASK_TO_VARIANTS: Dict[Task, Tuple[Variant, ...]] = {
    Task.SWEEP_TO_TOP: (
        Variant.DEMO,
        Variant.LAYOUT,
        Variant.SHAPE,
        Variant.COLOR,
        Variant.DYNAMICS,
        Variant.ALL,
    ),
}
EMBODIMENT_TO_CLASS: Dict[Embodiment, Callable] = {
    Embodiment.GRIPPER: embodiments.NonHolonomicGripperEmbodiment,
    Embodiment.SHORTSTICK: embodiments.ShortstickEmbodiment,
    Embodiment.MEDIUMSTICK: embodiments.MediumstickEmbodiment,
    Embodiment.LONGSTICK: embodiments.LongstickEmbodiment,
}
VARIANT_TO_KWARG: Dict[Variant, str] = {
    # Variant.DEMO need not set any kwarg.
    Variant.JITTER: "rand_layout_minor",
    Variant.LAYOUT: "rand_layout_full",
    Variant.COLOR: "rand_colors",
    Variant.SHAPE: "rand_shapes",
    Variant.DYNAMICS: "rand_dynamics",
    # Variant.ALL needs to set all rand_* kwargs to True. We'll handle it
    # dynamically below.
}

# ======================================= #


@dataclasses.dataclass
class EnvConfig:
    task: Task
    embodiment: Embodiment
    obs_type: ObservationSpace
    view_mode: ViewMode
    variant: Variant
    version: str

    @property
    def env_name(self) -> str:
        """Return a human-friendly string variable."""
        strs = [
            self.task.value,
            self.embodiment.value,
            self.obs_type.value,
            self.view_mode.value,
            self.variant.value,
            self.version,
        ]
        return "-".join(strs)

    @classmethod
    def from_name(cls, env_name: str) -> "EnvConfig":
        parsed = env_name.split("-")
        assert len(parsed) == 6
        return cls(
            Task(parsed[0]),
            Embodiment(parsed[1]),
            ObservationSpace(parsed[2]),
            ViewMode(parsed[3]),
            Variant(parsed[4]),
            parsed[5],
        )

    @property
    def is_test(self) -> bool:
        return self.variant != Variant.DEMO

    @property
    def demo_env_name(self) -> str:
        strs = [
            self.task.value,
            self.embodiment.value,
            self.obs_type.value,
            self.view_mode.value,
            Variant.DEMO.value,
            self.version,
        ]
        return "-".join(strs)


DEFAULT_RES = (384, 384)
_REGISTERED = False
DEMO_ENVS_TO_TEST_ENVS_MAP = collections.OrderedDict()
ENV_TO_EMBODIMENTS_MAP = collections.OrderedDict()  # TODO(kevin): Fill this.
ALL_REGISTERED_ENVS = []


def register_envs() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return False
    _REGISTERED = True

    sweep_to_top_ep_len = 100
    sweep_to_top_configs = [
        (
            # Longstick agent requires little time to solve the task so we
            # shorten the episode length to make RL training converge faster.
            sweep_to_top_ep_len // 2
            if embodiment == Embodiment.LONGSTICK
            else sweep_to_top_ep_len,
            EnvConfig(
                Task.SWEEP_TO_TOP,
                embodiment,
                ObservationSpace.PIXELS,
                ViewMode.ALLOCENTRIC,
                variant,
                "v0",
            ),
        )
        for embodiment, variant in itertools.product(
            Embodiment, TASK_TO_VARIANTS[Task.SWEEP_TO_TOP]
        )
    ]

    # Collection of ALL env specifications.
    env_configs = [
        *sweep_to_top_configs,
    ]

    # These are common to all environments, no matter the config.
    common_kwargs = dict(
        res_hw=DEFAULT_RES,
        fps=8,
        phys_steps=10,
        phys_iter=10,
    )

    # Register all the envs and record their names.
    for episode_len, config in env_configs:
        ALL_REGISTERED_ENVS.append(config.env_name)

        # Default randomization args. Trim out any kwargs associated with a
        # variant that isn't supported by the task, as specified in
        # TASK_TO_VARIANTS.
        valid_variant_kwargs = {
            k: v
            for k, v in VARIANT_TO_KWARG.items()
            if k in TASK_TO_VARIANTS[config.task]
        }
        if config.variant == Variant.ALL:
            env_kwargs = {k: True for k in valid_variant_kwargs.values()}
        else:
            env_kwargs = {k: False for k in valid_variant_kwargs.values()}
            if config.variant in valid_variant_kwargs:
                env_kwargs[valid_variant_kwargs[config.variant]] = True
        env_kwargs["robot_cls"] = EMBODIMENT_TO_CLASS[config.embodiment]

        gym.register(
            config.env_name,
            entry_point=TASK_TO_EPOINT[config.task],
            kwargs={
                "max_episode_steps": episode_len,
                **common_kwargs,
                **env_kwargs,
            },
        )

        # Allocentric view variant for pixel observation space.
        env_name_ego = config.env_name.replace(
            ViewMode.ALLOCENTRIC.value,
            ViewMode.EGOCENTRIC.value,
        )
        gym.register(
            env_name_ego,
            entry_point=TASK_TO_EPOINT[config.task],
            kwargs={
                "max_episode_steps": episode_len,
                "view_mode": "ego",
                **common_kwargs,
                **env_kwargs,
            },
        )
        ALL_REGISTERED_ENVS.append(env_name_ego)

        # Register STATE observation env if available.
        if TASK_TO_STATE_AVAILABILITY[config.task]:
            env_name_state = config.env_name.replace(
                ObservationSpace.PIXELS.value,
                ObservationSpace.STATE.value,
            )
            gym.register(
                env_name_state,
                entry_point=TASK_TO_EPOINT[config.task],
                kwargs={
                    "max_episode_steps": episode_len,
                    "use_state": True,
                    **common_kwargs,
                    **env_kwargs,
                },
            )
            ALL_REGISTERED_ENVS.append(env_name_state)

            # Allocentric view variant for pixel observation space.
            env_name_state_ego = env_name_state.replace(
                ViewMode.ALLOCENTRIC.value,
                ViewMode.EGOCENTRIC.value,
            )
            gym.register(
                env_name_state_ego,
                entry_point=TASK_TO_EPOINT[config.task],
                kwargs={
                    "max_episode_steps": episode_len,
                    "use_state": True,
                    "view_mode": "ego",
                    **common_kwargs,
                    **env_kwargs,
                },
            )
            ALL_REGISTERED_ENVS.append(env_name_state_ego)

    train_to_test_map = {}
    observed_demo_envs = set()
    for name in ALL_REGISTERED_ENVS:
        parsed = EnvConfig.from_name(name)
        if parsed.is_test:
            test_l = train_to_test_map.setdefault(parsed.demo_env_name, [])
            test_l.append(parsed.env_name)
        else:
            observed_demo_envs.add(name)

    # Use immutable values.
    train_to_test_map = {k: tuple(v) for k, v in train_to_test_map.items()}

    envs_with_test_variants = train_to_test_map.keys()
    assert observed_demo_envs == envs_with_test_variants, (
        "There are some train envs without test envs, or test envs without "
        "train envs"
    )
    sorted_items = sorted(train_to_test_map.items())
    DEMO_ENVS_TO_TEST_ENVS_MAP.update(sorted_items)

    return True
