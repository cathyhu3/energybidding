import torch
from torch import nn
from torch import optim
from torch.optim import Adam

from collections import namedtuple, deque
import random
import math
from copy import deepcopy

# Default Hyperparameter values
BATCH_SIZE = 32        # the number of experiences sampled from replay buffer
GAMMA = 0.92            # discount factor when calculating Q-value (high = thinking in long-term)
EPS_START = 0.9         # starting value of epsilon (action exploration constant)
EPS_END = 0.05          # end value of epsilon
EPS_DECAY = 1000        # rate of decay for epsilon
TAU = 0.05               # update rate of target network (the amount at which we keep update to the source params) -- tau << 1 to keep stable
LR = 1e-3               # learning rate of optimizer (used in GD)
EP_BF = 50              # number of episodes to wait before training
TOTAL_COST_CONST = 0    # set to 0: currently not using total_cost of the system as a factor in the loss function of each agent's Critic NN

class MADDPG:
    def __init__(self, market, batchsize, lr, gamma, eps_decay, tau, episodes_before_train=EP_BF, eps_start=EPS_START, eps_end=EPS_END):
        """
        Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

        hyperparameters (args):
            batch_size:             the number of experiences sampled from replay buffer
            gamma:                  discount factor when calculating Q-value (high = thinking in long-term)
            eps_start:              starting value of epsilon (action exploration constant)
            eps_end:                end value of epsilon
            eps_decay:              rate of decay for epsilon
            tau:                    update rate of target network (tau << 1 to keep target params stable)
            lr:                     learning rate of optimizer (used in GD)
            episodes_before_train:  the number of episodes stored in memory before we can start training
        
        """
        self.market = market # the agents and their corresponding actors, critics, memory replay buffers are stored in the market
        n_agents = len(market.agents)

        # hyperparameters
        self.batch_size = batchsize
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.episodes_before_train = episodes_before_train

        self.episodes = 0
        self.steps = 0
        self.trained_episodes = 0

        # determine which gpu programming software to use
        self.device = (
            "cuda"
            if torch.cuda.is_available() else "mps"
            if torch.backends.mps.is_available() else "cpu"
        )

        def calc_critic_input_dims(agent, market):
            pros, cons = market.num_pros_cons
            if agent.producing:
                return 9 + 4*(pros-1) + 2*cons
            return 7 + 4*pros + 2*(cons-1)

        # setting up Neural Networks for all agents
        for agent in market.agents:
            agent.critic = Critic(calc_critic_input_dims(agent, market), self.device).to(self.device)
            agent.target_critic = deepcopy(agent.critic).to(self.device)
            agent.critic_optimizer = Adam(agent.critic.parameters(), lr=self.lr)
            if agent.producing:
                agent.actor = ProsumerActor(self.device).to(self.device)
            else:
                agent.actor = ConsumerActor(self.device).to(self.device)
            agent.target_actor = deepcopy(agent.actor).to(self.device)
            agent.actor_optimizer = Adam(agent.actor.parameters(), lr=self.lr)

    def soft_update(self, target, source, tau):
        """
        Updates the parameters of the target network, given the parameters of the source network
            update rate = tau
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                        (1 - tau) * target_param.data + tau * source_param.data)
            
    def use_policy(self, agent, obs, batchsize=None, grad=True):
        """
        Use the policy:
            policy(state) = action
            actor(obs) = action
        Returns an action
        """
        if grad:
            return agent.actor(obs)
        with torch.no_grad():
            return agent.actor(obs)

    def select_action(self, agent, tou, fit):
        """
        Select an action based on the state:
            - using Policy (1-eps_thresh) % of the time
            - using random action eps_thresh % of the time
        
        action = [price decision, energy decision]
        """
        sample = random.random()
        eps_thresh = EPS_END + (EPS_START - EPS_END) * math.exp(-self.steps/EPS_DECAY)
        self.steps += 1
        if sample > eps_thresh:
            obs = torch.tensor(agent.get_observations(tou, fit), device=self.device)
            return self.use_policy(agent, obs, grad=False)
        else:
            price_decision = random.random()
            energy_decision = random.random()

            if agent.producing:
                # random buy and sell prices and quantities
                return torch.tensor([price_decision, energy_decision, random.random(), random.random()], device=self.device)
            return torch.tensor([price_decision, energy_decision], device=self.device)


    def update_parameters(self):
        """
        Update all of the network parameters (both the Critic and the Actor)
          for all of the agents from a randomly sampled batch
        (one step of training)

        Records the loss of the actor and critic
        """
        # don't train until enough exploration
        if self.episodes <= self.episodes_before_train:
            return None
        
        for agent in self.market.agents:
            experiences = agent.memory.sample(self.batch_size)
            batch = Episode(*zip(*experiences))
            def process_batch(b):
                """
                Transform all of the categories (orderbooks, obs, action, next_obs, reward, total_cost) of the batch of batchsize N
                  into flattened tensors of size (N, flattened length)
                
                (flattens all individual categories)
                """
                orderbook, observation, action, next_observation, reward, total_cost = b
                # individual processing
                orderbook = [torch.flatten(torch.stack(
                    [torch.stack(action) for order in batch for action in order] # stacking all of the actions within each order to 2 element tensors
                    )) for batch in orderbook] # stacking all of the orders witiin the a batch, then flattening them
                observation = torch.stack(observation)
                action = torch.stack(action)
                next_observation = torch.stack(next_observation)
                reward = torch.stack(reward)
                total_cost = torch.stack(total_cost)
                return [orderbook, observation, action, next_observation, reward, total_cost]

            def critic_input_processing(observation, action, orderbook):
                """
                Concatenate all of these critic inputs (obs, action, orderbook) into one flattened tensor
                """
                return torch.stack([
                    torch.cat((observation[i], action[i], orderbook[i])) 
                        for i in range(self.batch_size)])

            # processing batch to get input values
            orderbook, obs, action, next_obs, reward, total_cost = process_batch(batch)

            # formatted critic input
            critic_input = critic_input_processing(obs, action, orderbook)

            # formatted target critic input
            target_action = self.use_policy(agent, next_obs, self.batch_size)
            target_critic_input = critic_input_processing(next_obs, target_action, orderbook)
            input_size = critic_input.size(1)

            # Critic output (Q-value)
            Q_value = agent.critic(critic_input)
            y = reward + self.gamma*agent.target_critic(target_critic_input) + TOTAL_COST_CONST*total_cost

            # Critic Loss
            Q_loss = nn.MSELoss()(Q_value, y.detach()) # detach detaches result from gradient calculation
            agent.critic_loss.append(Q_loss)
            agent.critic_optimizer.zero_grad()
            Q_loss.backward()
            agent.critic_optimizer.step()

            # Actor Loss:
            actions = agent.actor(obs)
            critic_input = critic_input_processing(obs, actions, orderbook)
            policy_loss = agent.critic(critic_input).mean()
            agent.actor_loss.append(policy_loss)
            agent.actor_optimizer.zero_grad()
            policy_loss.backward()
            agent.actor_optimizer.step()

            self.trained_episodes += self.batch_size

            self.soft_update(agent.target_actor, agent.actor, self.tau)
            self.soft_update(agent.target_critic, agent.critic, self.tau)

Episode = namedtuple('Episode',
                        ('orderbook','obs', 'action', 'next_obs', 'reward', 'total_cost'))

class ReplayMemory:
    """ 
    Experience Replay Memory Buffer:
        contains episode tuples
        epsiode = ('obs', 'action', 'next_obs', 'reward (individual cost)', 'total cost') 
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        Adding an experience into memory
        """
        self.memory.append(Episode(*args))

    def sample(self, batch_size):
        """
        Retrieving a random sample of (batch_size) experiences from the ReplayMemory Buffer
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Critic(nn.Module):
    """
    Action-Value (Q function): neural network representation 

    input:
        [observation, action, ToU, FiT, sell-order books]
    output:
        expected future reward (cost) (of taking specific action given observation)
    """
    def __init__(self, in_dims, device, latent1=6, latent2=4,):
        super(Critic, self).__init__()
        # the types of NN layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dims, in_dims//2, device=device),                    # keep arbitrary 1 layer for now
            # nn.ReLU(),
            nn.BatchNorm1d(in_dims//2),
            nn.Linear(in_dims//2, 1, device=device),
        )
        self.device = device

    def forward(self, x):   # implements the neural network
        """
        x = input (total observation) = [observation, action, ToU, Fit, sell-order books]

        flattened = 3 + 2 + 2 + n_agents - 1 = 6 + n_agents
        """
        x.to(self.device)
        output = self.linear_relu_stack(x)
        return output

class ConsumerActor(nn.Module):
    """
    Policy neural network representation

    input:
        total observation = [observation, ToU, FiT] where observation = [inflexible demand, ES content, buffer]
        flattened = 3 + 1 + 1 = 5

    output: 
        action = [buy price decision, buy energy decision]
    """
    def __init__(self, device):
        super(ConsumerActor, self).__init__()
        # the types of NN layers
        self.linear_relu_stack = nn.Sequential( # nn.Sequential is an ordered container of modules
            nn.Linear(5, 4),                    # keep arbitrary 1 layer for now
            nn.ReLU(),
            nn.Linear(4, 2),
            # nn.ReLU(),
            # nn.Linear(3,2),
            nn.Sigmoid()
        )
        self.device = device

    def forward(self, x):   # implements the neural network
        """
        x = input = [inflexible demand, ES content, buffer, ToU, Fit]
        """
        x.to(self.device)
        logits = self.linear_relu_stack(x)
        logits.to(self.device)
        return logits
    
class ProsumerActor(nn.Module):
    """
    Policy neural network representation

    input:
        total observation = [observation, ToU, FiT]
        flattened = 3 + 1 + 1 = 5

    output:
        action = [sell price decision, sell energy decision, buy price decision, buy energy decision]
    """
    def __init__(self, device):
        super(ProsumerActor, self).__init__()
        # the types of NN layers
        self.linear_relu_stack = nn.Sequential( # nn.Sequential is an ordered container of modules
            nn.Linear(5, 4),                    # keep arbitrary 1 layer for now
            nn.ReLU(),
            # nn.Linear(5, 4),
            # nn.ReLU(),
            # nn.Linear(5, 1),
            # nn.Linear(4,4),
            nn.Sigmoid()
        )
        self.device = device

    def forward(self, x):   # implements the neural network
        """
        x = input = [inflexible demand, ES content, buffer, ToU, Fit]
        """
        x.to(self.device)
        logits = self.linear_relu_stack(x)
        logits.to(self.device)
        return logits


# -------------- TESTING -------------- #
# ReplayMemory test
# mem = ReplayMemory(4000)
# mem.push(1, 2, 3, 4, 5)
# mem.push(2, 4, 3, 4, 5)
# mem.push(0, 0, 0, 0, 0)
# sample = mem.sample(5)
# print(sample)