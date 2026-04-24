from evaluate.evaluate_agent import evaluate_agent
from evaluate.evaluate_baseline import evaluate_fixed_time_baseline

def compare_rl_vs_baseline(episodes=100, timesteps=50):
    print("=" * 60)
    print("COMPARING RL AGENT vs FIXED-TIME BASELINE")
    print("=" * 60)
    
    print("\nEvaluating Fixed-Time Baseline (VAC)...")
    baseline_results = evaluate_fixed_time_baseline(episodes=episodes, timesteps=timesteps)
    
    print("\nEvaluating RL Agent...")
    rl_results = evaluate_agent(load_path="q_table.json", episodes=episodes, timesteps=timesteps)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'RL Agent':>12} {'Baseline':>12} {'Winner':>12}")
    print("-" * 60)
    
    # Fairness (higher is better - Jain's index)
    rl_better_fairness = rl_results["avg_fairness"] > baseline_results["avg_fairness"]
    winner = "RL" if rl_better_fairness else "Baseline"
    print(f"{'Fairness (Jain)':<25} {rl_results['avg_fairness']:>12.3f} {baseline_results['avg_fairness']:>12.3f} {winner:>12}")
    
    # Throughput (higher is better)
    rl_better_throughput = rl_results["avg_throughput"] > baseline_results["avg_throughput"]
    winner = "RL" if rl_better_throughput else "Baseline"
    print(f"{'Throughput':<25} {rl_results['avg_throughput']:>12.2f} {baseline_results['avg_throughput']:>12.2f} {winner:>12}")
    
    # Waiting time (lower is better)
    rl_better_waiting = rl_results["avg_waiting"] < baseline_results["avg_waiting"]
    winner = "RL" if rl_better_waiting else "Baseline"
    print(f"{'Waiting Time (lower)':<25} {rl_results['avg_waiting']:>12.2f} {baseline_results['avg_waiting']:>12.2f} {winner:>12}")
    
    # Summary
    rl_wins = sum([rl_better_fairness, rl_better_throughput, rl_better_waiting])
    print("\n" + "=" * 60)
    if rl_wins >= 2:
        print(f"RESULT: RL WINS ({rl_wins}/3 metrics)")
    elif rl_wins == 1:
        print(f"RESULT: TIED ({rl_wins}/3 metrics)")
    else:
        print(f"RESULT: BASELINE WINS ({3-rl_wins}/3 metrics)")
    print("=" * 60)

if __name__ == "__main__":
    compare_rl_vs_baseline(episodes=100, timesteps=50)