from evaluate.evaluate_agent import evaluate_agent
from evaluate.evaluate_baseline import evaluate_fixed_time_baseline
from utils.plot_comparison import plot_comparison


def run_comparison():
    print("\nRunning RL Evaluation...")
    rl_results = evaluate_agent()

    print("\nRunning Baseline Evaluation...")
    baseline_results = evaluate_fixed_time_baseline()

    plot_comparison(rl_results, baseline_results)


if __name__ == "__main__":
    run_comparison()