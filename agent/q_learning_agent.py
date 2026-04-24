import random
import json
import math

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99,
                 alpha_decay=1.0, use_double_q=True, use_softmax=False, temperature=1.0):
        self.actions = actions
        self.alpha = alpha
        self.initial_alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay
        self.use_double_q = use_double_q
        self.use_softmax = use_softmax
        self.temperature = temperature
        
        # Double Q-Learning uses two Q-tables
        if use_double_q:
            self.q_table_1 = {}
            self.q_table_2 = {}
        else:
            self.q_table = {}
        
        self.training_step = 0

    def _get_q_table(self):
        """Get the primary Q-table for action selection"""
        if self.use_double_q:
            return self.q_table_1
        return self.q_table

    def _get_q_value(self, state, action):
        """Get Q-value from appropriate table(s)"""
        if self.use_double_q:
            # Average of both tables
            q1 = self.q_table_1.get(state, {}).get(action, 0)
            q2 = self.q_table_2.get(state, {}).get(action, 0)
            return (q1 + q2) / 2
        return self.q_table.get(state, {}).get(action, 0)

    def _init_state(self, state):
        """Initialize a state in Q-table(s)"""
        q_init = {a: 0 for a in self.actions}
        if self.use_double_q:
            if state not in self.q_table_1:
                self.q_table_1[state] = q_init.copy()
            if state not in self.q_table_2:
                self.q_table_2[state] = q_init.copy()
        else:
            if state not in self.q_table:
                self.q_table[state] = q_init.copy()

    def choose_action(self, state):
        self._init_state(state)

        if self.use_softmax:
            return self._softmax_action(state)
        
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self._greedy_action(state)

    def _greedy_action(self, state):
        """Choose action with highest Q-value"""
        if self.use_double_q:
            q1 = self.q_table_1.get(state, {})
            q2 = self.q_table_2.get(state, {})
            # Combine Q-values
            combined = {a: q1.get(a, 0) + q2.get(a, 0) for a in self.actions}
            return max(combined, key=combined.get)
        return max(self.q_table[state], key=self.q_table[state].get)

    def _softmax_action(self, state):
        """Choose action using Boltzmann/softmax exploration"""
        if self.use_double_q:
            q1 = self.q_table_1.get(state, {})
            q2 = self.q_table_2.get(state, {})
            q_values = [q1.get(a, 0) + q2.get(a, 0) for a in self.actions]
        else:
            q_values = [self.q_table[state][a] for a in self.actions]
        
        # Compute softmax probabilities
        exp_q = [math.exp(q / self.temperature) for q in q_values]
        sum_exp = sum(exp_q)
        probs = [e / sum_exp for e in exp_q]
        
        # Sample from distribution
        r = random.random()
        cumulative = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return self.actions[i]
        return self.actions[-1]

    def update_q(self, state, action, reward, next_state):
        self._init_state(next_state)
        self.training_step += 1

        if self.use_double_q:
            self._update_double_q(state, action, reward, next_state)
        else:
            self._update_single_q(state, action, reward, next_state)

    def _update_single_q(self, state, action, reward, next_state):
        """Standard Q-learning update"""
        old_value = self.q_table[state][action]
        best_future = max(self.q_table[next_state].values())
        self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * best_future - old_value)

    def _update_double_q(self, state, action, reward, next_state):
        """Double Q-Learning update - reduces overestimation bias"""
        # Randomly select which Q-table to update
        if random.random() < 0.5:
            # Update Q1 using Q2 for bootstrap
            old_value = self.q_table_1[state][action]
            best_action = max(self.q_table_2[next_state], key=self.q_table_2[next_state].get)
            best_future = self.q_table_2[next_state][best_action]
            self.q_table_1[state][action] = old_value + self.alpha * (reward + self.gamma * best_future - old_value)
        else:
            # Update Q2 using Q1 for bootstrap
            old_value = self.q_table_2[state][action]
            best_action = max(self.q_table_1[next_state], key=self.q_table_1[next_state].get)
            best_future = self.q_table_1[next_state][best_action]
            self.q_table_2[state][action] = old_value + self.alpha * (reward + self.gamma * best_future - old_value)

    def decay_epsilon(self):
        """Exponential epsilon decay"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def decay_alpha(self):
        """Learning rate decay over training"""
        if self.alpha > 0.01:  # Minimum learning rate
            self.alpha = self.initial_alpha * math.exp(-0.0001 * self.training_step)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            if self.use_double_q:
                data = {
                    "q_table_1": {str(k): v for k, v in self.q_table_1.items()},
                    "q_table_2": {str(k): v for k, v in self.q_table_2.items()},
                    "use_double_q": True
                }
            else:
                data = {
                    "q_table": {str(k): v for k, v in self.q_table.items()},
                    "use_double_q": False
                }
            json.dump(data, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if data.get("use_double_q", False):
            self.use_double_q = True
            self.q_table_1 = {eval(k): v for k, v in data["q_table_1"].items()}
            self.q_table_2 = {eval(k): v for k, v in data["q_table_2"].items()}
        else:
            self.use_double_q = False
            self.q_table = {eval(k): v for k, v in data["q_table"].items()}

    def save_hyperparameters(self, filepath):
        hyperparams = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "alpha_decay": self.alpha_decay,
            "use_double_q": self.use_double_q,
            "use_softmax": self.use_softmax,
            "temperature": self.temperature
        }
        with open(filepath, 'w') as f:
            json.dump(hyperparams, f)

    def load_hyperparameters(self, filepath):
        with open(filepath, 'r') as f:
            hyperparams = json.load(f)
        self.alpha = hyperparams.get("alpha", self.alpha)
        self.gamma = hyperparams.get("gamma", self.gamma)
        self.epsilon = hyperparams.get("epsilon", self.epsilon)
        self.epsilon_min = hyperparams.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = hyperparams.get("epsilon_decay", self.epsilon_decay)
        self.alpha_decay = hyperparams.get("alpha_decay", self.alpha_decay)
        self.use_double_q = hyperparams.get("use_double_q", self.use_double_q)
        self.use_softmax = hyperparams.get("use_softmax", self.use_softmax)
        self.temperature = hyperparams.get("temperature", self.temperature)