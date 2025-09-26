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


# container sayısı ve arrival_rate'e müdahale ediyoruz, avg_cpu_util'i okuyoruz.
# State vektöründe sıfırıncı index hangisiydi, besinci index neydi gibi seylerden kaçmak için
# mumkun mertebe dict kullandım, en son torch'a girerken tensor'a çeviriyorum.
STATE_KEYS = ['num_containers', 'cpu_allocation', 'avg_cpu_util', 'workload_process_rate', 'workload_change_rate']
# DQN'in replay memory'si için transition tanımı
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

NUM_EXPERIMENTS = 1
# Episode sayisi
NUM_EPISODES = 1000
# Step sayisi
NUM_STEPS = 16
# Tekrarlanabilirlik için random seed
RANDOM_SEED = 42
# Container sayisi, min, max, ve initial
MIN_NUM_CONTAINERS = 1
MAX_NUM_CONTAINERS = 3

MIN_CPU_ALLOC = 40
MAX_CPU_ALLOC = 49

MAX_ARRIVAL_RATE = 30
# Aksiyon: kaç konteyner varsa o kadar aksiyon var
NUM_ACTIONS = (MAX_CPU_ALLOC - MIN_CPU_ALLOC + 1) * (MAX_NUM_CONTAINERS - MIN_NUM_CONTAINERS + 1)
# State: STATE_KEYS'teki eleman sayısı kadar. Network'e key sırası ile giriyor.
NUM_STATES = len(STATE_KEYS)
# Replay memory'den batch_size kadar random örnek alıp train ediyoruz.
BATCH_SIZE = 64
# replay memory'nin max büyüklüğü. Yeni bir transition eklenirse en eski siliniyor. (deque ile yapmışlar)
REPLAY_MEMORY_SIZE = 4000
# Bunu cok bilmiyorum, DQN training sırasında kullanılıyor.
GAMMA = 0.99
# Epsilon-greedy için başlangıç, bitiş ve decay hızı
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000
# Kaç episode'da bir target network'ü policy network ile aynı yapalım.
TARGET_UPDATE = 20

def action_tensor_to_dict(action: torch.Tensor) -> Dict:
    '''Network'den tensor olarak çıkan sıfır indeksli aksiyonu dict'e çeviriyoruz.'''

    action_index = action.item()
    # action_index 0 ise MIN_NUM_CONTAINERS konteyner, 1 ise MIN_NUM_CONTAINERS+1 konteyner, ... diye gidiyor.
    cpu_action = action_index % (MAX_CPU_ALLOC - MIN_CPU_ALLOC + 1)
    replica_action = action_index // (MAX_CPU_ALLOC - MIN_CPU_ALLOC + 1)
    action_dict = {'vertical': cpu_action + MIN_CPU_ALLOC, 'scale_to': replica_action + MIN_NUM_CONTAINERS}
    return action_dict

def state_to_tensor(state: Dict) -> torch.Tensor:
    '''State dict'ini torch tensor'a çeviriyoruz. Burada listeyi ekstra bir [] içine koyuyoruz ki
    birden fazla state'i birleştirince batch dimension'i oluşsun.'''

    return torch.FloatTensor([list(state.values())])

def normalize_state(state: Dict) -> Dict:
    '''State normalizasyonunu özellikle dict üzerinde yapıyoruz ki sayısal indeksler neydi pesinde kosmayalim'''

    normalized_state = {k:0 for k in STATE_KEYS}
    normalized_state['num_containers'] = state['num_containers'] / MAX_NUM_CONTAINERS
    normalized_state['avg_cpu_util'] = state['avg_cpu_util']
    normalized_state['workload_change_rate'] = state['workload_change_rate']
    normalized_state['workload_process_rate'] = state['workload_process_rate']
    normalized_state['cpu_allocation'] = state['cpu_allocation']
    return normalized_state


def state_to_reward(state: Dict) -> float:
    beta = 1.2
    w = 0.8
    perf_reward = 0
    if state['response_time'] > 1500.0: # temp threshold values
        perf_reward = 1 - beta * np.exp(state['response_time'] / 4500) # without normalization, this value explodes

    cost_reward = 0
    if state['avg_cpu_util'] < 0.8: # temp threshold values
        cost_reward = 1 - np.exp(1 - state['avg_cpu_util'])

    return w * perf_reward + (1 - w) * cost_reward

def print_step_info(episode: int, step: int, state: Dict, action: Dict, next_state: Dict, reward: float, loss: float, exploration: bool):
    '''Diagnostik amaçlı çeşitli bilgileri yazdırıyoruz.'''

    if not hasattr(print_step_info, 'count'):
        print_step_info.count = 0

    def p_keys(d: Dict) -> str:
        print('\t'.join([f'{k:>{len(k)}}' for k, v in d.items()]))
        print('\t'.join(['-'*len(k) for k, v in d.items()]))


    def p_values(d: Dict) -> str:
        print('\t'.join([f'{v:{len(k)}.3f}' if isinstance(v, float) else f'{v:{len(k)}d}' for k, v in d.items()]))


    all_dicts = {'episode': episode, 'step': step}
    for k, v in state.items():
        all_dicts[k] = v
    all_dicts['horizontal_action'] = action['scale_to']
    all_dicts['vertical_action'] = action['vertical']
    for k, v in next_state.items():
        all_dicts[f'next_{k}'] = v
    all_dicts['reward'] = reward
    all_dicts['policy_net_loss'] = loss
    all_dicts['exploration'] = exploration

    if print_step_info.count % NUM_STEPS == 0:
        p_keys(all_dicts)
    p_values(all_dicts)

    print_step_info.count += 1


class ReplayMemory(object):
    '''deque ile süper basit bir replay memory implementasyonu
    yapmışlar çok beğendim.'''

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
        '''Normalize edilmiş state batch vektörünü alıyor. Aksiyon skorlarını döndürüyor.
        B x num_states -> B x num_actions şeklinde.'''
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
        '''DQN implementasyonu, çok bilmiyorum ama birbirinin aynısı
        iki network kullanması enteresan. Bir bakarsınız hocam.'''
        self.env = env

        self.num_states = num_states
        self.num_actions = num_actions

        self.policy_net = DQNNetwork(num_states, num_actions)
        self.target_net = DQNNetwork(num_states, num_actions)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Target net'te hiçbir zaman train yapmyacağız.
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)

        self.steps_completed = 0

    def select_action(self, state: Dict) -> Tuple[torch.Tensor, bool]:
        '''Explore-exploit için epsilon-greedy stratejisi kullanıyoruz.'''
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_completed / EPS_DECAY)
        self.steps_completed += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # state dictionary'sini son anda normalize edip tensor'a çeviriyoruz.
                state_tensor = state_to_tensor(normalize_state(state))
                action = self.policy_net(state_tensor).argmax().view(1, 1)
                exploration = False
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)
            exploration = True

        # donen action sıfır indeksli.
        return action, exploration

    def select_greedy_action(self, state: Dict) -> Tuple[torch.Tensor, bool]:
        with torch.no_grad():
            # state dictionary'sini son anda normalize edip tensor'a çeviriyoruz.
            state_tensor = state_to_tensor(normalize_state(state))
            action = self.policy_net(state_tensor).argmax().view(1, 1)
            exploration = False

        return action, exploration

    def optimize_model(self):
        '''Bu kisma çok hakim değilim, siz bir goz atarsanız sevinirim.'''

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

    def train(self, num_episodes, num_steps):
        rewards = np.zeros((num_episodes, num_steps))
        losses = np.zeros((num_episodes, num_steps))

        for episode in range(num_episodes):
            state = self.env.reset()

            for step in range(num_steps):
                action, exploration = self.select_action(state)
                action_to_execute = action_tensor_to_dict(action)

                # Episode'u yarıda bırakma durumu yok, o yuzden done burada bosta
                next_state, reward, done = self.env.step(action_to_execute)

                # Eğitimde kullanmak için verileri replay memory'e ekliyoruz.
                self.memory.push(state_to_tensor(normalize_state(state)),
                                 action,
                                 state_to_tensor(normalize_state(next_state)),
                                 torch.tensor([reward]))

                # BATCH_SIZE kadar dolmayanan kadar eğitim yapamıyoruz.
                # Sonrasında her step'de optimize çalışacak. Elde edilen loss'u
                # kaydediyorum.
                if len(self.memory) >= BATCH_SIZE:
                    loss = self.optimize_model()
                else:
                    loss = math.nan

                # Diagnostic amaçlı çeşitli bilgileri yazdırıyoruz.
                print_step_info(episode, step, state, action_to_execute, next_state, reward, loss, exploration)
                rewards[episode, step] = reward
                losses[episode, step] = loss

                # next_state bir sonraki adımın state'i olacak. O anlamda state'e aslında action
                # almadan önceki state diyebiliriz.
                state = next_state

            # Target network'ü policy network ile aynı yapıyoruz.
            if episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        return {'rewards': rewards, 'losses': losses}

class RLEnvironment:

    def __init__(self):
        '''Bağımsız değişkenlerimiz bunlar.'''
        self.num_containers = None
        self.cpu_allocation = None
        self.workload_indx = None

        self.nsfw_data = pd.read_csv('../data/nsfw_experiment4.csv')

        self.workload = [
            14, 7, 5, 4, 3, 4, 3, 3, 2, 2, 3, 4, 6, 10, 14, 18, 21, 22, 24, 25,
            27, 27, 27, 26, 27, 27, 27, 26, 25, 24, 25, 24, 23, 22, 22, 22,
            23, 23, 24, 24, 26, 27, 30, 29, 26, 23, 19, 17
        ]


    def get_rl_states(self) -> Dict:
        filtered_data = self.nsfw_data.query(f'num_containers == {self.num_containers} and cpu == {self.cpu_allocation} and arrival_rate == {self.workload[self.workload_indx]} and prev_arrival_rate == {self.workload[self.workload_indx-1]}')

        assert not filtered_data.empty, (
            f"No data found for num_containers={self.num_containers}, "
            f"cpu={self.cpu_allocation}, "
            f"arrival_rate={self.workload[self.workload_indx]}, "
            f"prev_arrival_rate={self.workload[self.workload_indx]-1}"
        )

        sample = filtered_data.sample(1)
        avg_cpu_util = sample['avg_cpu_util'].values[0]
        response_time = sample['response_time'].values[0]
        instant_tps = sample['instant_tps'].values[0]

        # state contains an extended amount of information, but few of these features are given to the agent
        state = {}
        state['num_containers'] = self.num_containers
        state['workload_change_rate'] = self.workload[self.workload_indx] / self.workload[self.workload_indx-1]
        state['cpu_allocation'] = self.cpu_allocation
        state['avg_cpu_util'] = avg_cpu_util
        state['response_time'] = response_time
        state['workload_process_rate'] = instant_tps / self.workload[self.workload_indx]
        state['arrival_rate'] = self.workload[self.workload_indx]
        state['prev_arrival_rate'] = self.workload[self.workload_indx-1]
        return state

    def reset(self) -> Dict:
        '''Her episode başında environment'ı resetliyoruz.'''
        self.num_containers = random.choice(range(MIN_NUM_CONTAINERS, MAX_NUM_CONTAINERS+1))
        self.cpu_allocation = random.choice(range(MIN_CPU_ALLOC, MAX_CPU_ALLOC+1))
        self.workload_indx = random.randint(1, len(self.workload) - 1)

        state= self.get_rl_states()

        return state

    def reset_to(self, num_containers: int, cpu_allocation: float, workload_indx: int) -> Dict:
        '''Her episode başında environment'ı resetliyoruz.'''
        self.num_containers = num_containers
        self.cpu_allocation = cpu_allocation
        self.workload_indx = workload_indx

        state= self.get_rl_states()

        return state

    def step(self, action: Dict) -> Tuple[Dict, float, bool]:
        # Execute the action
        self.num_containers = action['scale_to']
        self.cpu_allocation = action['vertical']

        # rewind to first workload
        self.workload_indx += 1
        if self.workload_indx >= len(self.workload):
            self.workload_indx = 0

        # Observe the next state
        state = self.get_rl_states()

        # calculate the reward
        reward = state_to_reward(state)

        # check if done
        done = False

        return state, reward, done

def plot_history(history, save_path = None):
    num_episodes = history['rewards'].shape[0]

    episode_rewards =np.mean(history['rewards'], axis=1)
    smoothed_episode_rewards = np.zeros_like(episode_rewards)
    for i in range(num_episodes):
        start_idx = max(0, i - 10)
        smoothed_episode_rewards[i] = np.nanmean(episode_rewards[start_idx:i+1])

    episode_losses = np.mean(history['losses'], axis=1)
    smoothed_episode_losses = np.zeros_like(episode_losses)
    for i in range(num_episodes):
        start_idx = max(0, i - 10)
        smoothed_episode_losses[i] = np.mean(episode_losses[start_idx:i+1])

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax1, ax2 = axes.flatten()

    color = 'tab:red'
    ax1.set_xlabel('episode')
    ax1.set_ylabel('reward', color=color)
    ax1.plot(episode_rewards, color=color)
    ax1.plot(smoothed_episode_rewards, '--', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(episode_losses, color=color)
    ax2.plot(smoothed_episode_losses, '--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if save_path:
        file_path = Path(save_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saved reward/loss plot {save_path}')
        plt.savefig(save_path)


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


def test_policy(policy, env, n_episodes=20, metrics_to_track=["num_containers", "cpu_allocation", "avg_cpu_util", "arrival_rate", "response_time"]):
    print("Testing the policy...")
    episode_rewards = []
    episode_metrics = []  # e.g., cpu_usage, response_time, etc.

    for ep in range(n_episodes):
        state = env.reset_to(num_containers=2, cpu_allocation=42, workload_indx=0)
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
    plt.savefig(f'../results/testing_reward_exp{NUM_EXPERIMENTS}_eps{NUM_EPISODES}.pdf')
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
        plt.savefig(f'../results/testing_{metric_name}_exp{NUM_EXPERIMENTS}_eps{NUM_EPISODES}.pdf')
        plt.close()   # close figure to avoid overlap when looping



if __name__ == "__main__":
    '''Ana programımız budur.'''

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    episode_histories = []
    for exp in range(NUM_EXPERIMENTS):
        env = RLEnvironment()

        policy = DQN(env, NUM_STATES, NUM_ACTIONS)

        episode_history = policy.train(NUM_EPISODES, NUM_STEPS)
        episode_histories.append(episode_history)

        if exp == 0:
            test_policy(policy, env)

    plot_multi_history(episode_histories, save_path=f'../results/experiment04_exp{NUM_EXPERIMENTS}_eps{NUM_EPISODES}.pdf')
