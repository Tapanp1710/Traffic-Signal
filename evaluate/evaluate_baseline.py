from env.traffic_env import TrafficEnvironment
from utils.metrics import calculate_fairness


def evaluate_fixed_time_baseline(episodes=100, timesteps=50):
    env = TrafficEnvironment()
    fairness_scores = []
    waiting_times = []

    avg_queue_lengths = []
    max_queue_seen_global = 0

    for episode in range(episodes):
        state = env.reset()

        total_queue = 0
        max_queue_seen = 0

        for time in range(timesteps):
            action = (time // 3) % 4
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

    print(f"\n=== BASELINE (Fixed-Time) Results ===")
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
    evaluate_fixed_time_baseline()