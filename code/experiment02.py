from collections import namedtuple, deque
import random
import math
import numpy as np
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# container sayısı ve arrival_rate'e müdahale ediyoruz, avg_cpu_usage'ı okuyoruz.
# State vektöründe sıfırıncı index hangisiydi, besinci index neydi gibi seylerden kaçmak için
# mumkun mertebe dict kullandım, en son torch'a girerken tensor'a çeviriyorum.
STATE_KEYS = ['num_containers', 'arrival_rate', 'avg_cpu_usage']
# DQN'in replay memory'si için transition tanımı
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# Episode sayisi
NUM_EPISODES = 100
# Step sayisi
NUM_STEPS = 16
# Tekrarlanabilirlik için random seed
RANDOM_SEED = 42
# Container sayisi, min, max, ve initial
MIN_NUM_CONTAINERS = 1
MAX_NUM_CONTAINERS = 3
INITIAL_NUM_CONTAINERS = 1
# Aksiyon: kaç konteyner varsa o kadar aksiyon var
NUM_ACTIONS = MAX_NUM_CONTAINERS - MIN_NUM_CONTAINERS + 1
# State: STATE_KEYS'teki eleman sayısı kadar. Network'e key sırası ile giriyor.
NUM_STATES = len(STATE_KEYS)
# Guya uygulama gelen yükü temsil ediyor. Tamsayı olarak tanımladım.
MIN_ARRIVAL_RATE = 2
MAX_ARRIVAL_RATE = 2
INITIAL_ARRIVAL_RATE = 2
# Replay memory'den batch_size kadar random örnek alıp train ediyoruz.
BATCH_SIZE = 8
# replay memory'nin max büyüklüğü. Yeni bir transition eklenirse en eski siliniyor. (deque ile yapmışlar)
REPLAY_MEMORY_SIZE = 100
# Bunu cok bilmiyorum, DQN training sırasında kullanılıyor.
GAMMA = 0.999
# Epsilon-greedy için başlangıç, bitiş ve decay hızı
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
# Kaç episode'da bir target network'ü policy network ile aynı yapalım.
TARGET_UPDATE = 20

def action_tensor_to_dict(action: torch.Tensor) -> Dict:
    '''Network'den tensor olarak çıkan sıfır indeksli aksiyonu dict'e çeviriyoruz.'''

    action_index = action.item()
    # action_index 0 ise MIN_NUM_CONTAINERS konteyner, 1 ise MIN_NUM_CONTAINERS+1 konteyner, ... diye gidiyor.
    action_dict = {'scale_to': action_index + MIN_NUM_CONTAINERS}
    return action_dict

def state_to_tensor(state: Dict) -> torch.Tensor:
    '''State dict'ini torch tensor'a çeviriyoruz. Burada listeyi ekstra bir [] içine koyuyoruz ki 
    birden fazla state'i birleştirince batch dimension'i oluşsun.'''

    return torch.FloatTensor([list(state.values())])

def normalize_state(state: Dict) -> Dict:
    '''State normalizasyonunu özellikle dict üzerinde yapıyoruz ki sayısal indeksler neydi pesinde kosmayalim'''

    normalized_state = state.copy()
    normalized_state['num_containers'] = state['num_containers'] / MAX_NUM_CONTAINERS
    normalized_state['arrival_rate'] = state['arrival_rate'] / MAX_ARRIVAL_RATE
    return normalized_state


def state_to_reward(state: Dict) -> float:
    '''avg_cpu_usage ne kadar yüksekse o kadar iyi olsun isteyelim.'''
    reward = state['avg_cpu_usage']
    return reward

def print_step_info(episode: int, step: int, state: Dict, action: Dict, next_state: Dict, reward: float, loss: float, exploration: bool):
    '''Diganostik amaçlı çeşitli bilgileri yazdırıyoruz.'''

    if not hasattr(print_step_info, 'count'):
        print_step_info.count = 0

    # print the keys with formatting length equal to key lengt

    def p_keys(d: Dict) -> str:
        print(' '.join([f'{k:>{len(k)}}' for k, v in d.items()]))
        print(' '.join(['-'*len(k) for k, v in d.items()]))
        
                        
    def p_values(d: Dict) -> str:
        print(' '.join([f'{v:{len(k)}.3f}' if isinstance(v, float) else f'{v:{len(k)}d}' for k, v in d.items()]))


    all_dicts = {'episode': episode, 'step': step} 
    for k, v in state.items():
        all_dicts[k] = v
    all_dicts['action'] = action['scale_to']
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

    def __init__(self, num_containers: int, arrival_rate: float):
        '''Bağımsız değişkenlerimiz bunlar.'''        
        self.num_containers = num_containers
        self.arrival_rate = arrival_rate

        self.nsfw_data = pd.read_csv('../data/nsfw_experiment2.csv')


    def get_rl_states(self) -> Dict:
        '''Guya metrik topluyoruz burada. Şimdilik data içinden random sample alarak üretiyorum.
        Datadan bakılınca avg_cpu_usage ile num_container arasında ters orantı var.
        Reward da zaten avg_cpu_usage. Dolayısı ile agent num_container'ı azaltmaya çalışacak.'''
        
        # Data'yi süzdükten sonra random sample aliyoruz. Experiment02 özelinde süzdükten sonra her 
        # zaman veri kalacagini biliyorum o yuzden ek kontrol koymadim.
        filtered_data = self.nsfw_data.query(f'num_containers == {self.num_containers} and arrival_rate == {self.arrival_rate}')
        avg_cpu_usage = filtered_data.sample(1)['avg_cpu_usage'].values[0]

        # Butun state bilgilerini içeren bir dict oluşturuyoruz.
        # Torch'a dokunmadıkça state'i hep dictionary ve normalize etmeden tutuyorum.
        state = {k: 0 for k in STATE_KEYS}
        state['num_containers'] = self.num_containers
        state['arrival_rate'] = self.arrival_rate
        state['avg_cpu_usage'] = avg_cpu_usage
        return state

    def reset(self) -> Dict:
        '''Her episode başında environment'ı resetliyoruz.'''
        self.num_containers = random.choice(range(MIN_NUM_CONTAINERS, MAX_NUM_CONTAINERS+1))
        self.arrival_rate = INITIAL_ARRIVAL_RATE # Veride sadece bir arrival_rate var

        state = self.get_rl_states()
        return state 

    def step(self, action: Dict) -> Tuple[Dict, float, bool]:
        '''Step atmada hiç kontrol etmeden aksiyonu doğrudan uyguluyorum.'''

        # Execute the action
        self.num_containers = action['scale_to']
        
        # Observe the next state
        state = self.get_rl_states()

        # calculate the reward
        reward = state_to_reward(state)

        # check if done
        done = False

        return state, reward, done

def plot_history(history):
    import matplotlib.pyplot as plt
    num_episodes = history['rewards'].shape [0]

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
    ax1.plot(smoothed_episode_rewards, '-', color='black')
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(episode_losses, color=color)
    ax2.plot(smoothed_episode_losses, '-', color='black')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

if __name__ == "__main__":
    '''Ana programımız budur.'''

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    env = RLEnvironment(INITIAL_NUM_CONTAINERS, INITIAL_ARRIVAL_RATE)

    agent = DQN(env, NUM_STATES, NUM_ACTIONS)

    history = agent.train(NUM_EPISODES, NUM_STEPS)

    plot_history(history)

    print('Bitti!')
    print(f'"action" sütunu {MIN_NUM_CONTAINERS} sayısına yakınsamış olmalı!')
    print(f'Eğer exploration varsa rastgele bir değer görebilirsiniz.')