import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_env import TrafficEnvironment
from agent.q_learning_agent import QLearningAgent
from utils.metrics import calculate_fairness, calculate_throughput


def evaluate_agent(load_path="q_table.json", episodes=100, timesteps=50):
    env = TrafficEnvironment()
    agent = QLearningAgent(actions=[0, 1, 2, 3])

    # ✅ AUTO-DETECT MODEL PATH (no more confusion)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    possible_paths = [
        os.path.join(base_dir, "q_table.json"),
        os.path.join(base_dir, "train", "q_table.json"),
        os.path.join(base_dir, "models", "q_table.json"),
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError(
            f"q_table.json not found in expected locations:\n{possible_paths}"
        )

    print(f"✅ Loading model from: {model_path}")
    agent.load(model_path)

    fairness_scores = []
    throughput_scores = []
    waiting_times = []

    # ✅ NEW METRICS
    avg_queue_lengths = []
    max_queue_seen_global = 0

    for episode in range(episodes):
        state = env.reset()
        cars_passed = 0
        total_queue = 0
        max_queue_seen = 0

        for t in range(timesteps):
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            state = next_state

            # Track throughput
            if env.queue[env.current_green] < env.prev_queue[env.current_green]:
                cars_passed += 1

            # Track queue stats
            current_total_queue = sum(env.queue)
            total_queue += current_total_queue
            max_queue_seen = max(max_queue_seen, max(env.queue))

        # Episode metrics
        fairness = calculate_fairness(env.queue)
        throughput = calculate_throughput(cars_passed)

        fairness_scores.append(fairness)
        throughput_scores.append(throughput)
        waiting_times.append(env.total_waiting_time)

        avg_queue_lengths.append(total_queue / timesteps)
        max_queue_seen_global = max(max_queue_seen_global, max_queue_seen)

    # Final averages
    avg_fairness = sum(fairness_scores) / len(fairness_scores)
    avg_throughput = sum(throughput_scores) / len(throughput_scores)
    avg_waiting = sum(waiting_times) / len(waiting_times)
    avg_queue_length = sum(avg_queue_lengths) / len(avg_queue_lengths)

    # ✅ OUTPUT
    print(f"\n=== RL AGENT Results ===")
    print(f"Fairness (Jain's Index): {avg_fairness:.3f}")
    print(f"Throughput: {avg_throughput:.2f}")
    print(f"Waiting Time: {avg_waiting:.2f}")
    print(f"Average Queue Length: {avg_queue_length:.2f}")
    print(f"Max Queue Length Observed: {max_queue_seen_global}")

    return {
        "avg_fairness": avg_fairness,
        "avg_throughput": avg_throughput,
        "avg_waiting": avg_waiting,
        "avg_queue_length": avg_queue_length,
        "max_queue": max_queue_seen_global,
    }


if __name__ == "__main__":
    evaluate_agent()