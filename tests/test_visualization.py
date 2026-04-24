import unittest
from utils.visualization import plot_learning_curve, plot_queue_dynamics

class TestVisualization(unittest.TestCase):
    def test_plot_learning_curve(self):
        rewards = [i for i in range(10)]
        try:
            plot_learning_curve(rewards, title="Test Learning Curve")
        except Exception as e:
            self.fail(f"plot_learning_curve raised an exception: {e}")

    def test_plot_queue_dynamics(self):
        queue_lengths = [[i for i in range(10)] for _ in range(4)]
        try:
            plot_queue_dynamics(queue_lengths, title="Test Queue Dynamics")
        except Exception as e:
            self.fail(f"plot_queue_dynamics raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()