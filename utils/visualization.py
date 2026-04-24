import matplotlib.pyplot as plt

def plot_learning_curve(rewards, title="Learning Curve"):
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.show()

def plot_queue_dynamics(queue_lengths, title="Queue Dynamics"):
    for i, queue in enumerate(queue_lengths):
        plt.plot(queue, label=f"Road {i}")
    plt.xlabel("Time Steps")
    plt.ylabel("Queue Length")
    plt.title(title)
    plt.legend()
    plt.show()