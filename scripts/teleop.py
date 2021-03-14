import gym
from absl import app, flags

from xmagical import register_envs
from xmagical.utils import KeyboardEnvInteractor

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "env_name",
    "SweepToTop-Gripper-State-Allo-Demo-v0",
    "The environment to load.",
)
flags.DEFINE_boolean("exit_on_done", False, "Whether to exit if done is True.")


def main(_):
    register_envs()
    env = gym.make(FLAGS.env_name)
    viewer = KeyboardEnvInteractor(action_dim=env.action_space.shape[0])

    env.reset()
    obs = env.render("rgb_array")
    viewer.imshow(obs)

    i = [0]

    def step(action):
        obs, rew, done, info = env.step(action)
        if obs.ndim != 3:
            obs = env.render("rgb_array")
        if done and FLAGS.exit_on_done:
            return
        if i[0] % 100 == 0:
            print(f"Done, score {info['eval_score']:.2f}/1.00")
        i[0] += 1
        return obs

    viewer.run_loop(step)


if __name__ == "__main__":
    app.run(main)
