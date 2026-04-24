import random

# convert queue to state
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

# Q-table (empty dictionary)
Q = {}

# possible actions (roads)
actions = [0, 1, 2, 3]

# example queue
queue = [2, 5, 1, 7]

state = get_state(queue)

# initialize Q values if state not seen before
if state not in Q:
    Q[state] = {a: 0 for a in actions}

print("Queue:", queue)
print("State:", state)
print("Q-table entry:", Q[state])

epsilon = 0.2

def choose_action(state):
    # explore
    if random.random() < epsilon:
        return random.choice(actions)
    # exploit (choose best action)
    else:
        return max(Q[state], key=Q[state].get)

action = choose_action(state)

print("Chosen action (road):", action)

alpha = 0.1
gamma = 0.9

# fake example reward (we will connect real waiting time later)
reward = -10  

# next state example
next_queue = [1, 4, 0, 6]
next_state = get_state(next_queue)

# initialize next state if not present
if next_state not in Q:
    Q[next_state] = {a: 0 for a in actions}

# Q-learning update
old_value = Q[state][action]
best_future = max(Q[next_state].values())

Q[state][action] = old_value + alpha * (reward + gamma * best_future - old_value)

print("Updated Q-value:", Q[state][action])
