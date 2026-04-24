import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpdf import FPDF
import matplotlib.pyplot as plt
from evaluate.evaluate_agent import evaluate_agent
from evaluate.evaluate_baseline import evaluate_fixed_time_baseline

def generate_pdf_report():
    # Evaluate RL Agent
    print("Evaluating RL Agent...")
    rl_results = evaluate_agent(load_path="q_table.json", episodes=100)

    # Evaluate Fixed-Time Baseline
    print("Evaluating Fixed-Time Baseline...")
    baseline_results = evaluate_fixed_time_baseline()

    # Create PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Traffic Signal Control Report", ln=True, align="C")

    # RL vs Baseline Metrics
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Comparison of RL Agent and Fixed-Time Baseline", ln=True, align="L")
    pdf.ln(10)

    metrics = ["Average Reward", "Average Fairness", "Average Throughput"]
    rl_values = [rl_results["avg_reward"], rl_results["avg_fairness"], rl_results["avg_throughput"]]
    baseline_values = [baseline_results["avg_reward"], baseline_results["avg_fairness"], baseline_results["avg_throughput"]]

    for metric, rl, baseline in zip(metrics, rl_values, baseline_values):
        pdf.cell(200, 10, txt=f"{metric}: RL = {rl}, Baseline = {baseline}", ln=True, align="L")

    # Add Plots
    pdf.add_page()
    pdf.cell(200, 10, txt="Learning Curve and Queue Dynamics", ln=True, align="C")

    # Example Plot
    plt.plot([1, 2, 3], [4, 5, 6])  # Replace with actual data
    plt.title("Example Plot")
    plt.savefig("example_plot.png")
    pdf.image("example_plot.png", x=10, y=50, w=190)

    # Save PDF
    pdf.output("Traffic_Signal_Control_Report.pdf")
    print("PDF report generated: Traffic_Signal_Control_Report.pdf")

if __name__ == "__main__":
    generate_pdf_report()