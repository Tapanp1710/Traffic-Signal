def calculate_fairness(queue_lengths):
    """
    Calculate fairness using Jain's Fairness Index.
    Higher is better (1.0 = perfect fairness).
    """
    n = len(queue_lengths)
    if n == 0:
        return 1.0
    
    total = sum(queue_lengths)
    if total == 0:
        return 1.0  # Perfect fairness when no queues
    
    sum_squared = sum(q ** 2 for q in queue_lengths)
    jain_index = (total ** 2) / (n * sum_squared) if sum_squared > 0 else 1.0
    return jain_index

def calculate_throughput(cars_passed):
    """
    Calculate throughput - number of cars that passed through the intersection.
    Higher is better.
    """
    return cars_passed