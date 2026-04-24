import random

queue = [0, 0, 0, 0]
current_green = 0
green_time = 0

for time in range(30):
    print("\nTime step:", time)

    # cars arrive
    for i in range(4):
        if random.random() < 0.5:
            queue[i] += 1

    # cars leave from green road
    if queue[current_green] > 0:
        queue[current_green] -= 1

    print("Green road:", current_green)
    print("Queues:", queue)

    # keep green for 3 steps
    green_time += 1
    if green_time == 3:
        current_green = (current_green + 1) % 4
        green_time = 0
