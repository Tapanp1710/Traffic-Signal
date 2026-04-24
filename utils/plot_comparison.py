import matplotlib.pyplot as plt


def plot_comparison(rl_results, baseline_results):

    # ================= WAITING TIME =================
    rl_wait = rl_results['avg_waiting']
    base_wait = baseline_results['avg_waiting']

    values = [rl_wait, base_wait]
    labels = ['RL Agent', 'Baseline']

    plt.figure()
    plt.bar(labels, values)
    plt.title("Waiting Time Comparison", fontsize=14)
    plt.ylabel("Waiting Time")

    # Add values on bars
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.1f}", ha='center')

    plt.savefig("waiting_time_comparison.png")
    plt.show()


    # ================= AVG QUEUE =================
    rl_queue = rl_results['avg_queue_length']
    base_queue = baseline_results['avg_queue_length']

    values = [rl_queue, base_queue]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Average Queue Length Comparison", fontsize=14)
    plt.ylabel("Average Queue Length")

    # Add values on bars
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.2f}", ha='center')

    plt.savefig("avg_queue_comparison.png")
    plt.show()


    # ================= FAIRNESS =================
    rl_fair = rl_results['avg_fairness']
    base_fair = baseline_results['avg_fairness']

    values = [rl_fair, base_fair]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Fairness Comparison", fontsize=14)
    plt.ylabel("Jain's Fairness Index")

    # Add values on bars
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha='center')

    plt.savefig("fairness_comparison.png")
    plt.show()