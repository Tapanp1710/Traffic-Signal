from env.traffic_env import TrafficEnvironment
from utils.metrics import calculate_fairness, calculate_throughput

def evaluate_fixed_time_baseline(episodes=100, timesteps=50):
    env = TrafficEnvironment()
    fairness_scores = []
    throughput_scores = []
    waiting_times = []

    for episode in range(episodes):
        state = env.reset()
        cars_passed = 0

        for time in range(timesteps):
            # Fixed-time policy: rotate green light every 3 steps
            action = (time // 3) % 4
            next_state, reward = env.step(action)
            state = next_state
            
            # Track cars passed (throughput)
            if env.queue[env.current_green] < env.prev_queue[env.current_green]:
                cars_passed += 1

        # Calculate metrics
        fairness = calculate_fairness(env.queue)
        throughput = calculate_throughput(cars_passed)
        waiting_times.append(env.total_waiting_time)
        fairness_scores.append(fairness)
        throughput_scores.append(throughput)

    avg_fairness = sum(fairness_scores) / len(fairness_scores)
    avg_throughput = sum(throughput_scores) / len(throughput_scores)
    avg_waiting = sum(waiting_times) / len(waiting_times)
    
    print(f"\n=== BASELINE (Fixed-Time) Results ===")
    print(f"Fairness (Jain's Index): {avg_fairness:.3f}")
    print(f"Throughput: {avg_throughput:.2f}")
    print(f"Waiting Time: {avg_waiting:.2f}")

    return {
        "avg_fairness": avg_fairness,
        "avg_throughput": avg_throughput,
        "avg_waiting": avg_waiting
    }

if __name__ == "__main__":
    evaluate_fixed_time_baseline()