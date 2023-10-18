from unittest import TestCase

from lunar_lander import LunarLanderEnvironment


class TestLunarLanderEnvironment(TestCase):
    def test_env_init(self):
        lunar_lander_env = LunarLanderEnvironment()
        lunar_lander_env.env_init()
