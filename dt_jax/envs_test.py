import test_with_cpu  # noqa: F401  # isort:skip
import random

from absl.testing import absltest
from envs import AtariEnv, AtariEnvConfig


class AtariEnvTest(absltest.TestCase):
    def test_atari_env_integration(self):
        config = AtariEnvConfig(seed=17, game="breakout")
        env = AtariEnv(config)
        env.reset()
        actions = list(env.actions.keys())
        for i in range(10):
            random_action = random.choice(actions)
            o, _, _ = env.step(random_action)
        self.assertEqual(o.shape, (4, 84, 84))


if __name__ == "__main__":
    absltest.main()
