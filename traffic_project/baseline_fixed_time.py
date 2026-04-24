import random

queue = [0, 0, 0, 0]
current_green = 0
green_time = 0
total_waiting_time = 0

for time in range(30):
    print("\nTime step:", time)

    # cars arrive
    for i in range(4):
        arrival_prob = 0.5

        if time >= 15 and i == 0:
            arrival_prob = 0.9

        if random.random() < arrival_prob:
            queue[i] += 1

    # cars leave from green road
    if queue[current_green] > 0:
        queue[current_green] -= 1

    # calculate waiting time
    for i in range(4):
        if i != current_green:
            total_waiting_time += queue[i]

    print("Green road:", current_green)
    print("Queues:", queue)
    print("Total waiting time so far:", total_waiting_time)

    # keep green for 3 steps
    green_time += 1
    if green_time == 3:
        current_green = (current_green + 1) % 4
        green_time = 0
