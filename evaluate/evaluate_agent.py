import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_env import TrafficEnvironment
from agent.q_learning_agent import QLearningAgent
from utils.metrics import calculate_fairness


def evaluate_agent(load_path="q_table.json", episodes=100, timesteps=50):
    env = TrafficEnvironment()
    agent = QLearningAgent(actions=[0, 1, 2, 3])

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, load_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"✅ Loading model from: {model_path}")
    agent.load(model_path)

    fairness_scores = []
    waiting_times = []

    avg_queue_lengths = []
    max_queue_seen_global = 0

    for episode in range(episodes):
        state = env.reset()

        total_queue = 0
        max_queue_seen = 0

        for t in range(timesteps):
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            state = next_state

            current_total_queue = sum(env.queue)
            total_queue += current_total_queue
            max_queue_seen = max(max_queue_seen, max(env.queue))

        fairness = calculate_fairness(env.queue)

        fairness_scores.append(fairness)
        waiting_times.append(env.total_waiting_time)

        avg_queue_lengths.append(total_queue / timesteps)
        max_queue_seen_global = max(max_queue_seen_global, max_queue_seen)

    avg_fairness = sum(fairness_scores) / len(fairness_scores)
    avg_waiting = sum(waiting_times) / len(waiting_times)
    avg_queue_length = sum(avg_queue_lengths) / len(avg_queue_lengths)

    print(f"\n=== RL AGENT Results ===")
    print(f"Fairness (Jain's Index): {avg_fairness:.3f}")
    print(f"Waiting Time: {avg_waiting:.2f}")
    print(f"Average Queue Length: {avg_queue_length:.2f}")
    print(f"Max Queue Length Observed: {max_queue_seen_global}")

    return {
        "avg_fairness": avg_fairness,
        "avg_waiting": avg_waiting,
        "avg_queue_length": avg_queue_length,
        "max_queue": max_queue_seen_global,
    }


if __name__ == "__main__":
    evaluate_agent()