import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from evaluate.evaluate_agent import evaluate_agent
from evaluate.evaluate_baseline import evaluate_fixed_time_baseline

def generate_summary_report():
    print("Evaluating RL Agent...")
    rl_results = evaluate_agent(load_path="q_table.json", episodes=100)

    print("Evaluating Fixed-Time Baseline...")
    baseline_results = evaluate_fixed_time_baseline()

    # Write results to CSV
    with open("summary_report.csv", "w", newline="") as csvfile:
        fieldnames = ["Metric", "RL Agent", "Fixed-Time Baseline"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({"Metric": "Average Reward", "RL Agent": rl_results["avg_reward"], "Fixed-Time Baseline": baseline_results["avg_reward"]})
        writer.writerow({"Metric": "Average Fairness", "RL Agent": rl_results["avg_fairness"], "Fixed-Time Baseline": baseline_results["avg_fairness"]})
        writer.writerow({"Metric": "Average Throughput", "RL Agent": rl_results["avg_throughput"], "Fixed-Time Baseline": baseline_results["avg_throughput"]})

    print("Summary report generated: summary_report.csv")

if __name__ == "__main__":
    generate_summary_report()