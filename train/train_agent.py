import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from env.traffic_env import TrafficEnvironment
from agent.q_learning_agent import QLearningAgent

def evaluate_agent(env, agent, num_episodes=10, timesteps=50):
    """Evaluate agent performance without exploration"""
    total_waiting = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_waiting = 0
        
        for _ in range(timesteps):
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            episode_reward += reward
            episode_waiting += env.total_waiting_time
            state = next_state
        
        total_waiting.append(episode_waiting)
    
    return np.mean(episode_reward), np.mean(total_waiting)

def train_agent(episodes=5000, save_path="q_table.json", eval_interval=200, early_stop_patience=500):
    """
    Enhanced training with:
    - More episodes and timesteps
    - Learning rate decay
    - Evaluation during training
    - Early stopping based on convergence
    """
    env = TrafficEnvironment()
    
    # Optimized hyperparameters
    agent = QLearningAgent(
        actions=[0, 1, 2, 3],
        alpha=0.15,           # Slightly higher initial learning rate
        gamma=0.95,           # Higher discount for future rewards
        epsilon=1.0,
        epsilon_min=0.01,     # Lower minimum exploration
        epsilon_decay=0.995,  # Slower decay for more exploration
        alpha_decay=0.9999,   # Learning rate decay
        use_double_q=True,    # Use Double Q-Learning
        use_softmax=False     # Can toggle to True for softmax exploration
    )

    rewards = []           # Track rewards for visualization
    epsilon_values = []    # Track epsilon decay
    eval_rewards = []      # Track evaluation performance
    eval_waiting = []      # Track evaluation waiting time
    
    best_eval_reward = float('-inf')
    patience_counter = 0
    best_q_table = None
    
    timesteps = 60         # More timesteps per episode

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for time in range(timesteps):
            action = agent.choose_action(state)
            next_state, reward = env.step(action)

            agent.update_q(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)
        
        # Decay learning rate
        agent.decay_alpha()
        agent.decay_epsilon()

        # Evaluate periodically
        if episode % eval_interval == 0 and episode > 0:
            eval_reward, eval_wait = evaluate_agent(env, agent, num_episodes=5, timesteps=timesteps)
            eval_rewards.append(eval_reward)
            eval_waiting.append(eval_wait)
            
            print(f"Episode {episode}, Train Reward: {total_reward:.1f}, "
                  f"Eval Reward: {eval_reward:.1f}, Eval Waiting: {eval_wait:.1f}, "
                  f"Alpha: {agent.alpha:.4f}, Epsilon: {agent.epsilon:.4f}")
            
            # Early stopping check
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                patience_counter = 0
                # Save best model
                if agent.use_double_q:
                    best_q_table = (agent.q_table_1.copy(), agent.q_table_2.copy())
                else:
                    best_q_table = agent.q_table.copy()
            else:
                patience_counter += eval_interval
            
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at episode {episode}. Best eval reward: {best_eval_reward:.1f}")
                break
        elif episode % 500 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.1f}")

    # Restore best model if we have one
    if best_q_table is not None:
        if agent.use_double_q:
            agent.q_table_1, agent.q_table_2 = best_q_table
        else:
            agent.q_table = best_q_table

    agent.save(save_path)
    agent.save_hyperparameters("hyperparameters.json")
    print("Training complete. Q-table and hyperparameters saved.")

if __name__ == "__main__":
    train_agent()