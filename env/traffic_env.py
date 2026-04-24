import random

class TrafficEnvironment:
    def __init__(self, min_green=3, yellow_time=2, stochastic_arrivals=True, base_arrival_rate=0.5):
        self.queue = [0, 0, 0, 0]
        self.prev_queue = [0, 0, 0, 0]
        self.current_green = 0
        self.green_time = 0
        self.total_waiting_time = 0
        self.cars_passed = 0
        self.switch_count = 0
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.yellow_phase = False
        self.stochastic_arrivals = stochastic_arrivals
        self.base_arrival_rate = base_arrival_rate
        self.time_step = 0
        self.episode_queue_history = []

    def reset(self):
        self.queue = [0, 0, 0, 0]
        self.prev_queue = [0, 0, 0, 0]
        self.current_green = 0
        self.green_time = 0
        self.total_waiting_time = 0
        self.cars_passed = 0
        self.switch_count = 0
        self.time_step = 0
        self.episode_queue_history = []
        return self.get_state()

    def get_arrival_rate(self):
        if not self.stochastic_arrivals:
            return self.base_arrival_rate
        
        hour_factor = 0.3 + 0.4 * (1 + (self.time_step % 20 - 10) / 10)
        return min(0.8, self.base_arrival_rate * hour_factor)

    def step(self, action):
        self.time_step += 1
        self.prev_queue = self.queue.copy()

        # Initialize rewards
        waiting_penalty = 0
        throughput_bonus = 0
        queue_bonus = 0
        fairness_bonus = 0
        queue_control_bonus = 0
        target_road_bonus = 0
        clear_bonus = 0
        switch_penalty = 0

        if self.yellow_phase:
            self.yellow_phase = False
            return self.get_state(), 0

        if self.green_time < self.min_green:
            action = self.current_green

        switching = (action != self.current_green)

        if switching:
            self.yellow_phase = True
            self.green_time = 0
            self.switch_count += 1
            return self.get_state(), 0

        self.current_green = action
        self.green_time += 1

        # Cars departure
        cars_departed = 0
        if self.queue[self.current_green] > 0:
            self.queue[self.current_green] -= 1
            cars_departed = 1
        self.cars_passed += cars_departed

        # Arrivals
        arrival_rate = self.get_arrival_rate()
        for i in range(4):
            if random.random() < arrival_rate:
                self.queue[i] += 1

        # Waiting time
        waiting_time = sum(self.queue[i] for i in range(4) if i != self.current_green)
        self.total_waiting_time += waiting_time

        self.episode_queue_history.append(self.queue.copy())
        next_state = self.get_state()

        # ------------------ REWARD FUNCTION ------------------

        # Queue reduction bonus
        queue_reduction = sum(self.prev_queue) - sum(self.queue)
        queue_bonus = max(0, queue_reduction * 3)

        # Throughput bonus
        throughput_bonus = cars_departed * 45

        # ✅ NEW SAFE FAIRNESS (continuous, low impact)
        avg_q = sum(self.queue) / len(self.queue)
        imbalance = sum(abs(q - avg_q) for q in self.queue)

        # Lower imbalance → higher reward (scaled safely)
        fairness_bonus = max(0, 6 - imbalance * 0.5)

        # Waiting penalty (MOST IMPORTANT)
        waiting_penalty = waiting_time * 0.45

        # Target road bonus
        max_queue = max(self.queue)
        if action == self.queue.index(max_queue):
            target_road_bonus = 8

        # Queue control bonus
        total_queue = sum(self.queue)
        queue_control_bonus = max(0, (12 - total_queue) * 1.2)

        # Clear bonus
        if cars_departed > 0 and self.queue[self.current_green] == 0:
            clear_bonus = 12

        # Switch penalty
        if switching:
            if self.queue[action] < max_queue * 0.6:
                switch_penalty = 2

        reward = (
            -waiting_penalty
            + throughput_bonus
            + queue_bonus
            + fairness_bonus
            + queue_control_bonus
            + target_road_bonus
            + clear_bonus
            - switch_penalty
        )

        return next_state, reward

    def get_state(self):
        state = []

        for q in self.queue:
            if q == 0:
                state.append(0)
            elif q <= 2:
                state.append(1)
            elif q <= 4:
                state.append(2)
            elif q <= 7:
                state.append(3)
            elif q <= 12:
                state.append(4)
            else:
                state.append(5)

        state.append(min(self.green_time, 8))

        total_prev = sum(self.prev_queue)
        total_curr = sum(self.queue)

        if total_curr > total_prev + 3:
            state.append(2)
        elif total_curr > total_prev:
            state.append(1)
        elif total_curr < total_prev - 3:
            state.append(0)
        else:
            state.append(1)

        state.append(self.current_green)

        total_q = sum(self.queue)
        if total_q <= 3:
            state.append(0)
        elif total_q <= 8:
            state.append(1)
        elif total_q <= 15:
            state.append(2)
        else:
            state.append(3)

        return tuple(state)

    def render(self):
        print(f"Queues: {self.queue}, Current Green: {self.current_green}, Total Waiting Time: {self.total_waiting_time}")

    def get_queue_dynamics(self):
        return self.queue.copy()


class MultiIntersectionEnvironment:
    def __init__(self, num_intersections=2, min_green=3, yellow_time=2):
        self.num_intersections = num_intersections
        self.intersections = [TrafficEnvironment(min_green, yellow_time) for _ in range(num_intersections)]

    def reset(self):
        return [intersection.reset() for intersection in self.intersections]

    def step(self, actions):
        next_states = []
        rewards = []

        for i, action in enumerate(actions):
            next_state, reward = self.intersections[i].step(action)
            next_states.append(next_state)
            rewards.append(reward)

        return next_states, rewards

    def render(self):
        for i, intersection in enumerate(self.intersections):
            print(f"Intersection {i}:")
            intersection.render()