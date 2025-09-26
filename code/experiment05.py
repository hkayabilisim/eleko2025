from collections import namedtuple, deque
import random
import math
import numpy as np
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# State is stored in dictionary with the following attributes
STATE_KEYS = ['num_containers',
              'cpu_shares_per_container',
              'avg_cpu_usage',
              'cpu_utilization',
              'data_processing_rate',
              'arrival_rate',
              'previous_arrival_rate',
              'processing_rate',
              'arrival_change_rate',
              'latency']
# These keys of the state are used in the policy
STATE_KEYS_IN_POLICY = ['num_containers',
                          'cpu_shares_per_container',
                          'cpu_utilization',
                          'data_processing_rate',
                          'arrival_change_rate'
                          ]
# Replay memory of DQN uses transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# Number of experiments
NUM_EXPERIMENTS = 10
# Number of episodes
NUM_EPISODES = 1000
# Number of steps in each episode
NUM_STEPS = 16
# Random seed for reproducibility
RANDOM_SEED = 42
# Minimum and maximum number of containers
MIN_NUM_CONTAINERS = 1
MAX_NUM_CONTAINERS = 3
# CPU allocation in milicpus
MIN_CPU_SHARES_PER_CONTAINER = 4000
MAX_CPU_SHARES_PER_CONTAINER = 4900
INCR_CPU_SHARES_PER_CONTAINER = 100
# Maximum arrival rate (unit is number of requests per second)
MAX_ARRIVAL_RATE = 30
# Latency objective in seconds
SLO_LATENCY = 1.5
# Maximum latency observed in the data in seconds
MAX_LATENCY = 4.5
# Service Level Objective in CPU utilization
SLO_CPU_UTILIZATION = 0.8
# Horizontal/vertical actions
VERTICAL_ACTIONS = list(range(MIN_CPU_SHARES_PER_CONTAINER, MAX_CPU_SHARES_PER_CONTAINER + INCR_CPU_SHARES_PER_CONTAINER, INCR_CPU_SHARES_PER_CONTAINER))
HORIZONTAL_ACTIONS = list(range(MIN_NUM_CONTAINERS, MAX_NUM_CONTAINERS + 1))
# Number of actions
NUM_ACTIONS_HORIZONTAL = len(HORIZONTAL_ACTIONS)
NUM_ACTIONS_VERTICAL = len(VERTICAL_ACTIONS)
NUM_ACTIONS = NUM_ACTIONS_HORIZONTAL * NUM_ACTIONS_VERTICAL
# Number of states in the policy networks
NUM_STATES = len(STATE_KEYS_IN_POLICY)
# Batch size for neural network training. These are sampled from replay memory.
BATCH_SIZE = 64
# Replay memory size. Memory is implemented as a deque.
REPLAY_MEMORY_SIZE = 4000
# A hyperparameter in DQN training
GAMMA = 0.99
# Epsilon-greedy parameters
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000
# Target network is updated every <TARGET_UPDATE> episodes.
TARGET_UPDATE = 20

def action_tensor_to_dict(action: torch.Tensor) -> Dict:
    '''Converts the output of the DQN network to action dict.'''

    action_index = action.item()
    vertical_action = action_index % NUM_ACTIONS_VERTICAL
    horizontal_action = action_index // NUM_ACTIONS_VERTICAL
    action_dict = {'vertical': VERTICAL_ACTIONS[vertical_action],
                   'horizontal': HORIZONTAL_ACTIONS[horizontal_action]}
    return action_dict

def state_to_tensor(state: Dict) -> torch.Tensor:
    '''State dict is converted to torch tensor. We use extra [] around the state
    so that when accumulated a batch dimension is produced.'''

    return torch.FloatTensor([list(state.values())])

def state_subset(state: Dict) -> Dict:
    '''Take the subset of the state that goes to neural networks.'''

    return {k: state[k] for k in STATE_KEYS_IN_POLICY}

def normalize_state(state: Dict) -> Dict:
    '''State normalization is handled via dictionary keys 
    so that we don't need a separate bookkeeping for indexes.
    This function is only used right before feeding
    the state to the neural network.'''

    normalized_state = state.copy()
    normalized_state['num_containers'] /= MAX_NUM_CONTAINERS
    normalized_state['cpu_shares_per_container'] /=  MAX_CPU_SHARES_PER_CONTAINER
    normalized_state['avg_cpu_usage'] /= MAX_CPU_SHARES_PER_CONTAINER
    normalized_state['arrival_rate'] /= MAX_ARRIVAL_RATE
    normalized_state['previous_arrival_rate'] /= MAX_ARRIVAL_RATE
    normalized_state['processing_rate'] /= MAX_ARRIVAL_RATE

    return normalized_state


def state_to_reward(state: Dict) -> float:
    '''Calculate reward based on state. Note that input state is non-normalized'''
    beta = 1.0 / state['latency']
    w = 0.8
    perf_reward = 0
    if state['latency'] > SLO_LATENCY:
        perf_reward = 1 - np.exp(beta * state['latency'])

    cost_reward = 0
    if state['cpu_utilization'] < SLO_CPU_UTILIZATION:
        cost_reward = 1 - np.exp(1 - state['cpu_utilization'])

    return w * perf_reward + (1 - w) * cost_reward

def print_step_info(episode: int, step: int, state: Dict, action: Dict, next_state: Dict, reward: float, loss: float, exploration: bool):
    '''Printing diagnostics.'''

    if not hasattr(print_step_info, 'count'):
        print_step_info.count = 0

    def p_keys(d: Dict) -> str:
        '''Printing the keys of a dictionary.'''
        print(' '.join([f'{k:>{len(k)}}' for k, v in d.items()]))
        print(' '.join(['-'*len(k) for k, v in d.items()]))

    def p_values(d: Dict) -> str:
        '''Printing the values of a dictionary. Key widths are used.'''
        print(' '.join([f'{v:{len(k)}.3f}' if isinstance(v, float) else f'{v:{len(k)}d}' for k, v in d.items()]))

    all_dicts = {'episode': episode, 'step': step}
    for k, v in state.items():
        all_dicts[k] = v
    all_dicts['horizontal_action'] = action['horizontal']
    all_dicts['vertical_action'] = action['vertical']
    #for k, v in next_state.items():
    #    all_dicts[f'next_{k}'] = v
    all_dicts['reward'] = reward
    all_dicts['loss'] = loss
    all_dicts['exploration'] = exploration

    if print_step_info.count % NUM_STEPS == 0:
        p_keys(all_dicts)
    p_values(all_dicts)

    print_step_info.count += 1


class ReplayMemory(object):
    '''A replay memory implementation with deque'''

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        '''Takes normalized batch matrix and returns the action scores:
        B x num_states -> B x num_actions where B is batch size.'''
        super(DQNNetwork, self).__init__()

        self.fc1 = nn.Linear(num_states, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class DQN:
    def __init__(self, env, num_states, num_actions):
        '''DQN implementation.'''
        self.env = env

        self.num_states = num_states
        self.num_actions = num_actions

        self.policy_net = DQNNetwork(num_states, num_actions)
        self.target_net = DQNNetwork(num_states, num_actions)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Target net is never trained.
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)

        self.steps_completed = 0

    def select_action(self, state: Dict) -> Tuple[torch.Tensor, bool]:
        '''We use epsilon-greedy strategy for explore-exploit.'''
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_completed / EPS_DECAY)
        self.steps_completed += 1
        if sample > eps_threshold:
            action, exploration = self.select_greedy_action(state)
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)
            exploration = True

        # zero-index action is returned.
        return action, exploration

    def select_greedy_action(self, state: Dict) -> Tuple[torch.Tensor, bool]:
        with torch.no_grad():
            # state -> normalize -> take subset -> convert to tensor
            state_tensor = state_to_tensor(state_subset(normalize_state(state)))
            action = self.policy_net(state_tensor).argmax().view(1, 1)
            exploration = False

        return action, exploration

    def optimize_model(self):
        '''Optimization of the policy network.'''

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()

        loss = criterion(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes, num_steps, verbose = False):
        rewards = np.zeros((num_episodes, num_steps))
        losses = np.zeros((num_episodes, num_steps))

        for episode in range(num_episodes):
            state = self.env.reset()

            for step in range(num_steps):
                action, exploration = self.select_action(state)
                action_to_execute = action_tensor_to_dict(action)

                # Episode is never broken so 'done' is redundant here
                next_state, reward, done = self.env.step(action_to_execute)

                # We push state->action->next-state->reward transition to replay memory
                self.memory.push(state_to_tensor(state_subset(normalize_state(state))),
                                 action,
                                 state_to_tensor(state_subset(normalize_state(next_state))),
                                 torch.tensor([reward]))

                # optimize if only the replay memory has enough data
                if len(self.memory) >= BATCH_SIZE:
                    loss = self.optimize_model()
                else:
                    loss = math.nan

                if verbose:
                    print_step_info(episode, step, state, action_to_execute, next_state, reward, loss, exploration)

                # We store rewards and losses for episode,step pairs.
                rewards[episode, step] = reward
                losses[episode, step] = loss

                # next_state will be current state in the next iteration.
                state = next_state

            # Update target network periodically
            if episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        return {'rewards': rewards, 'losses': losses}

class RLEnvironment:

    def __init__(self, data_path=None):
        '''These variables are used to sample states from the offline data'''
        self.num_containers = None
        self.cpu_shares_per_container = None
        self.arrival_rate = None
        self.previous_arrival_rate = None
        self.workload_indx = None

        self.nsfw_data = pd.read_csv(data_path)

        # arrival_rate values
        self.workload = [
            14, 7, 5, 4, 3, 4, 3, 3, 2, 2, 3, 4, 6, 10, 14, 18, 21, 22, 24, 25,
            27, 27, 27, 26, 27, 27, 27, 26, 25, 24, 25, 24, 23, 22, 22, 22,
            23, 23, 24, 24, 26, 27, 30, 29, 26, 23, 19, 17
        ]


    def get_rl_states(self) -> Dict:
        '''Sampling from the offline data matching the four attributes'''

        filtered_data = self.nsfw_data.query(f'num_containers == {self.num_containers} and '+
                                             f'arrival_rate == {self.arrival_rate} and '+
                                             f'previous_arrival_rate == {self.previous_arrival_rate} and '+
                                             f'cpu_shares_per_container == {self.cpu_shares_per_container}')

        assert not filtered_data.empty, (
            f"No data found for num_containers={self.num_containers}, "
            f"cpu_shares_per_container={self.cpu_shares_per_container}, "
            f"arrival_rate={self.arrival_rate}, "
            f"previous_arrival_rate={self.previous_arrival_rate}"
        )

        sample = filtered_data.sample(1)
        # From the offline data, we only take the state attributes
        state = {k: sample[k].values[0] for k in STATE_KEYS}

        return state

    def reset_to(self, num_containers: int, cpu_shares_per_container: int, workload_indx: int) -> Dict:
        '''Reset to specific state.'''
        self.num_containers = num_containers
        self.cpu_shares_per_container = cpu_shares_per_container
        self.workload_indx = workload_indx

        self.arrival_rate = self.workload[self.workload_indx]
        self.previous_arrival_rate = self.workload[(self.workload_indx - 1) % len(self.workload)]

        state= self.get_rl_states()

        return state

    def reset(self) -> Dict:
        '''Reset to random state.'''

        num_containers = random.choice(HORIZONTAL_ACTIONS)
        cpu_shares_per_container = random.choice(VERTICAL_ACTIONS)
        workload_indx = random.randint(0, len(self.workload) - 1)

        return self.reset_to(num_containers, cpu_shares_per_container, workload_indx)

    def step(self, action: Dict) -> Tuple[Dict, float, bool]:
        # Execute the action
        self.num_containers = action['horizontal']
        self.cpu_shares_per_container = action['vertical']

        self.workload_indx += 1
        # prev/arrival_rate is taken from workload trace
        self.arrival_rate = self.workload[self.workload_indx % len(self.workload)]
        self.previous_arrival_rate = self.workload[(self.workload_indx - 1) % len(self.workload)]

        # Observe the next state
        state = self.get_rl_states()

        # calculate the reward
        reward = state_to_reward(state)

        # check if done
        done = False

        return state, reward, done


def plot_multi_history(histories, save_path=None):
    """
    Plot average rewards and losses across multiple experiments with std shading.

    Args:
        histories (list): list of dicts with keys {'rewards', 'losses'}.
                          Each 'rewards' and 'losses' is shaped (num_episodes, num_steps).
        save_path (str): optional path to save the figure.
    """
    # Collect episode totals
    all_rewards, all_losses = [], []

    for h in histories:
        rewards = np.sum(h['rewards'], axis=1)  # total reward per episode
        losses = np.mean(h['losses'], axis=1)  # avg loss per episode
        all_rewards.append(rewards)
        all_losses.append(losses)

    all_rewards = np.stack(all_rewards)  # (num_experiments, num_episodes)
    all_losses = np.stack(all_losses)

    num_experiments, num_episodes = all_rewards.shape

    # Mean and std across experiments
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    mean_losses = np.mean(all_losses, axis=0)
    std_losses = np.std(all_losses, axis=0)

    # Plot
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
    ax1, ax2 = axes.flatten()

    # Rewards
    ax1.set_title("Training Results Across Experiments")
    ax1.set_ylabel("Reward")
    ax1.plot(mean_rewards, color="tab:red", label="Mean Reward")
    ax1.fill_between(range(num_episodes),
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     color="tab:red", alpha=0.2, label="±1 std")
    ax1.legend()

    # Losses
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.plot(mean_losses, color="tab:blue", label="Mean Loss")
    ax2.fill_between(range(num_episodes),
                     mean_losses - std_losses,
                     mean_losses + std_losses,
                     color="tab:blue", alpha=0.2, label="±1 std")
    ax2.legend()

    fig.tight_layout()

    if save_path:
        file_path = Path(save_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saved reward/loss plot {save_path}")
        plt.savefig(save_path)


def test_policy(policy, env, n_episodes=20, prefix='experiment', metrics_to_track=["num_containers", "cpu_shares_per_container", "cpu_utilization", "arrival_rate", "latency"]):
    print("Testing the policy...")
    episode_rewards = []
    episode_metrics = []  # e.g., cpu_usage, response_time, etc.

    for ep in range(n_episodes):
        state = env.reset_to(num_containers=2, cpu_shares_per_container=4200, workload_indx=0)
        rewards = []
        metrics = []

        metrics.append({m: state[m] for m in metrics_to_track})

        for step in range(len(env.workload) * 3):
            action, _ = policy.select_greedy_action(state)
            action_to_execute = action_tensor_to_dict(action)
            next_state, reward, done = env.step(action_to_execute)

            # store metrics
            rewards.append(reward)
            metrics.append({m: next_state[m] for m in metrics_to_track})

            state = next_state
            if done:
                break

        episode_rewards.append(rewards)
        episode_metrics.append(metrics)

    # Convert to numpy for easier math
    rewards_array = np.array([np.array(r) for r in episode_rewards])  # shape (episodes, steps)
    mean_rewards = rewards_array.mean(axis=0)
    std_rewards = rewards_array.std(axis=0)

    timesteps = np.arange(rewards_array.shape[1])

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, mean_rewards, label="Mean Reward")
    plt.fill_between(timesteps,
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     alpha=0.2, label="±1 std")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(f'../results/{prefix}_testing_reward_exp{NUM_EXPERIMENTS}_eps{NUM_EPISODES}.pdf')
    plt.close()   # close figure to avoid overlap when looping

    # Extract cpu_usage from metrics
    for metric_name in metrics_to_track:
        value_array = np.array([[m[metric_name] for m in ep] for ep in episode_metrics])
        mean = value_array.mean(axis=0)
        std = value_array.std(axis=0)

        timesteps = np.arange(value_array.shape[1])

        plt.figure(figsize=(8, 5))
        plt.plot(timesteps, mean, label="App2Scale Agent")
        plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2)
        plt.xlabel("Timestep")
        plt.ylabel(metric_name)
        plt.legend()
        plt.savefig(f'../results/{prefix}_testing_{metric_name}_exp{NUM_EXPERIMENTS}_eps{NUM_EPISODES}.pdf')
        plt.close()   # close figure to avoid overlap when looping



if __name__ == "__main__":
    '''Ana programımız budur.'''

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    episode_histories = []
    for exp in range(NUM_EXPERIMENTS):
        env = RLEnvironment(data_path='../data/nsfw_experiment5.csv')

        policy = DQN(env, NUM_STATES, NUM_ACTIONS)

        episode_history = policy.train(NUM_EPISODES, NUM_STEPS, verbose=True)
        episode_histories.append(episode_history)

        if exp == 0:
            test_policy(policy, env, prefix='experiment05')

    plot_multi_history(episode_histories, save_path=f'../results/experiment05_exp{NUM_EXPERIMENTS}_eps{NUM_EPISODES}.pdf')
