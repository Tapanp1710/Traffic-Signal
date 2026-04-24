import unittest
from agent.q_learning_agent import QLearningAgent

class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QLearningAgent(actions=[0, 1, 2, 3])

    def test_choose_action(self):
        state = (0, 0, 0, 0)
        action = self.agent.choose_action(state)
        self.assertIn(action, [0, 1, 2, 3])

    def test_update_q(self):
        state = (0, 0, 0, 0)
        next_state = (1, 0, 0, 0)
        self.agent.update_q(state, 0, -1, next_state)
        self.assertIn(state, self.agent.q_table)
        self.assertIn(0, self.agent.q_table[state])

    def test_save_and_load(self):
        state = (0, 0, 0, 0)
        self.agent.update_q(state, 0, -1, state)
        self.agent.save("test_q_table.json")

        new_agent = QLearningAgent(actions=[0, 1, 2, 3])
        new_agent.load("test_q_table.json")
        self.assertEqual(new_agent.q_table, self.agent.q_table)

if __name__ == "__main__":
    unittest.main()