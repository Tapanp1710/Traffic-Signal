import random

# ----------------------------
# STATE REPRESENTATION
# ----------------------------
def get_state(queue):
    state = []
    for q in queue:
        if q <= 2:
            state.append(0)
        elif q <= 5:
            state.append(1)
        else:
            state.append(2)
    return tuple(state)

# ----------------------------
# Q-TABLE SETUP
# ----------------------------
Q = {}
actions = [0, 1, 2, 3]

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
alpha = 0.1
gamma = 0.9

def choose_action(state):
    global epsilon

    if state not in Q:
        Q[state] = {a: 0 for a in actions}

    if random.random() < epsilon:
        action = random.choice(actions)
    else:
        action = max(Q[state], key=Q[state].get)

    return action

def update_q(state, action, reward, next_state):
    if next_state not in Q:
        Q[next_state] = {a: 0 for a in actions}

    old_value = Q[state][action]
    best_future = max(Q[next_state].values())

    Q[state][action] = old_value + alpha * (reward + gamma * best_future - old_value)

# ----------------------------
# TRAINING LOOP
# ----------------------------
episodes = 2000

for episode in range(episodes):

    queue = [0, 0, 0, 0]
    total_waiting_time = 0

    for time in range(30):

        # vehicle arrival
        for i in range(4):
            arrival_prob = 0.5

            if time >= 15 and i == 0:
                arrival_prob = 0.9

            if random.random() < arrival_prob:
                queue[i] += 1

        # get state
        state = get_state(queue)

        # choose action
        action = choose_action(state)
        current_green = action

        # vehicle departure
        if queue[current_green] > 0:
            queue[current_green] -= 1

        # waiting time
        waiting = sum(queue[i] for i in range(4) if i != current_green)
        total_waiting_time += waiting

        # reward
        reward = -waiting

        # next state
        next_state = get_state(queue)

        # update Q-table
        update_q(state, action, reward, next_state)

    # print progress every 20 episodes
    if episode % 20 == 0:
        print(f"Episode {episode}, Total Waiting Time: {total_waiting_time}")

        # reduce exploration gradually
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("\nTraining complete.")