import unittest
from env.traffic_env import TrafficEnvironment

class TestTrafficEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = TrafficEnvironment()

    def test_reset(self):
        state = self.env.reset()
        self.assertEqual(state, (0, 0, 0, 0))
        self.assertEqual(self.env.total_waiting_time, 0)

    def test_step(self):
        self.env.reset()
        next_state, reward = self.env.step(0)
        self.assertEqual(len(next_state), 4)
        self.assertIsInstance(reward, int)

    def test_get_state(self):
        self.env.queue = [1, 4, 6, 0]
        state = self.env.get_state()
        self.assertEqual(state, (0, 1, 2, 0))

if __name__ == "__main__":
    unittest.main()