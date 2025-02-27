from maddpg import ReplayMemory
import torch

class Utility:
    def __init__(self, ToU, FiT):
        """
        ToU (Time of Use)= price that agents will pay for energy from utility company
        FiT (Feed in Tarrif) = the money given to agents that sell energy to utility
        """
        self.tou = ToU
        self.fit = FiT

class Market:
    """
    A market special-type which represents the specific energy market
    """
    def __init__(self, pros_cons, utility, agents=[]):
        """
        self.utility
            stores Utility (class) object
        self.agents
            stores all of the Agent objects in the market
        self.num_pros_cons
            tuple of (# of prosumers, # of consumers)
        
        ~~For Each Market Period~~
        self.orderbook
            the market input to be cleared
            [buy_orders = { buyer1 = [price1, quantity1]
                            buyer2 = [price2, quantity2]
                            ...
                            },
            sell_orders = { seller1 = [price1, quantity1]
                            seller2 = [price2, quantity2]
                            ...
                            }
            ]
        self.outcome
            stores the market outcome to be displayed in the simulation
            [
                {(seller, buyer): [quantity, price, total], ...},   hour 1
                {(seller, buyer): [quantity, price, total], ...},   hour 2
                ...
            ]
        self.total_cost
            keeps track of the total generation cost during a day of training
        self.daily_totalcosts
            keeps track of the daily total cost for every day of training
            (can track if daily generation cost goes down over steps of training)
        self.sim_total_cost
            keeps track of the total generation cost during the simulation
        
        self.device
            Indicates the device to store all of the PyTorch Tensors on
        """
        self.utility = utility
        self.agents = agents
        self.num_pros_cons = pros_cons # tuple of (# of prosumers, # of consumers)

        self.orderbook = self.init_orders()

        self.total_cost = 0
        self.daily_totalcosts = []

        self.outcome = {}
        self.sim_totalcost = []     # records total generation cost of the simulation
        self.orderbooks = []
        self.outcomes = []
        self.device = (
            "cuda"
            if torch.cuda.is_available() else "mps"
            if torch.backends.mps.is_available() else "cpu"
        )

    def get_ToU(self, hour):
        """
        Returns Time of Use (ToU) utility price
          for the specified hour of the day (0-23)
        """
        return self.utility.tou[hour]
    
    def get_FiT(self, hour):
        """
        Returns Feed in Tarrif (FiT) utility price
          for the specified hour of the day (0-23)
        """
        return self.utility.fit[hour]
    
    def tou_fit_inputform(self, hour, batchsize=None):
        """
        Returns a tou and fit representation used as
          part of the input into Critic and Actor NN

        tensor([
                tensor([tou, fit]),
                tensor([tou, fit]),
                ...
                ])
        """
        if batchsize:
            return torch.stack([torch.tensor((self.utility.tou[hour], self.utility.fit[hour]), device=self.device) for i in range(batchsize)])
        return torch.tensor((self.utility.tou[hour], self.utility.fit[hour]), device=self.device)
    
    def clear_all_energy(self):
        """
        Clears the energy balance of all agents
        (used at the beginning of simulation)
        """
        for agent in self.agents:
            agent.reset_energy_bal()

    def init_orders(self):
        """
        Initializes the format of the orderbook. 
        The orderbook consists of 2 dictionaries:
            - buy orders dict
            - sell orders dict
        where each buyer / seller's name is mapped 
        to the [price, quantity] they are willing to buy / sell at
            - the price and quantity are initialized to 0
        """
        buy_orders = {}
        sell_orders = {}
        for agent in self.agents:
            if agent.producing:
                sell_orders[agent.name] = [torch.tensor(0, device=self.device),torch.tensor(0, device=self.device)]
            buy_orders[agent.name] = [torch.tensor(0, device=self.device),torch.tensor(0, device=self.device)]
        return [buy_orders, sell_orders]

    def clear_all_money(self):
        for agent in self.agents:
            agent.reset_money()

    def get_agent(self, name):
        """
        Given the name of an agent, return that agent's instance from the market
            name: the name of the agent in a numerical representation
        """
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def add_agent(self, agent):
        self.agents.append(agent)

class ConsumingAgent:
    """
    Represents an Agent within the market
    """
    def __init__(self, name, sim_load):
        """
        ~~Unique Characteristics of Agent~~
        self.name
            a unique name (represented by a number > 0) which is given to each agent in the market
        self.producing
            a boolean representing if the agent is a prosumer or a consumer
        self.sim_load
            daily load customer data for a year
                [
                    [daily load data, 30 min increments],   day #0
                    ...
                    [daily load data, 30 min increments],   day #364
                ]

        ~~RL Attributes~~
        self.observation
            current observation of the agent: [inflexible_load, ES_content, buffer, tou, fit]
        self.past_obs
            stores past observation
        self.reward
            reward for the past market hour 
            (the net cost of the energy for the past market hour)

        ~~Reinforcement Learning DDPG Model of the Agent~~
        (all instantiated when a MADDPG object is created at the start of training)
        self.actor
            stores the Actor NN defined by maddpg.py
        self.critic
            stores the Critic NN defined by maddpg.py 
        self.target_actor + critic
            deepcopy of the actor / critic NNs used in the
            process of updating the Actor / Critic NN parameters
        self.policy / critic optimizer
            torch.optim optimizers userd in parameter updating
        self.memory
            replay memory buffer object which stores experiences of the agent 
            to be used in training:
            (past_obs, action, obs, reward)

        ~~Training Progress Characteristics~~
        self.critic_loss
            tracks critic loss over steps of training
        self.actor_loss
            tracks actor loss over steps of training

        ~~Simulation Characteristics to Track~~
        self.money
            keeps track of aggregate the money earned / lost from the market 
        self.energy_bal_record
            a record of the energy balance during the sim
        self.money_record
            a record of the money of the agent during the sim
        """
        self.name = name
        self.producing = False
        self.sim_load = sim_load

        self.observation = Observations()
        self.past_obs = None
        self.reward = 0 
        self.money = 0

        self.actor = None
        self.critic = None   
        self.target_actor = None
        self.target_critic = None
        self.policy_optimizer = None
        self.critic_optimizer = None
        self.memory = ReplayMemory(8000)

        self.critic_loss = []
        self.actor_loss = []

        self.energy_bal_record = []
        self.money_record = []

    def __str__(self):
        return f"Agent {self.name}"
    
    def get_observations(self, tou, fit):
        """
        Returns the observation of the agent
            self.observations = [inflexible load, ES content, Buffer]
        """
        return self.observation.get_obs(tou, fit)
    
    def get_energy_bal(self):
        """
        Returns the current "Energy Balance" of the Agent:
        the net energy balance of each agent per hour 
            (PV production + energy traded (lost or gained) from P2P market - energy used (load))
        """
        return self.observation.ES_content

    def determine_inflexible(self, day, ix):
        """
        Calculate the inflexible load based on agent load data, and data index
            The inflex load for the past hour would be the average of (ix, and ix - 1) in the dataset
        
        - used in updating the current observation
        """
        self.observation.inflexible_load = (self.sim_load[day][ix] + self.sim_load[day][ix-1])/2

    def add_energy_bal(self, val):
        """
        Add a value to the energy balance
        """
        self.observation.ES_content += val
        if self.producing:
            self.observation.buffer = self.ES_max - self.observation.ES_content
    
    def reset_energy_bal(self):
        """
        Resets the energy balance
        """
        self.observation.ES_content = 0
    
    def reset_money(self):
        """
        Resets money (used in simulation)
        """
        self.money = 0
    
class ProsumingAgent(ConsumingAgent):
    """
    Subclass of Agent class representing an agent who can produce and consume (has an Energy Storage and  PV production)
    """
    def __init__(self, name, ES_max, PV_prod_sim, sim_load):
        """
        ES_max:
            max ES capacity (everything is relative to ES min)
        PV_prod_sim:
            list of simulated PV production at every simulation second (30 mins) for a day
        sim_load:
            the simulated daily load at every simulation second (30 mins) for a day
        """
        super(ProsumingAgent, self).__init__(name, sim_load)
        self.producing = True
        self.ES_max = ES_max
        self.PV_prod_sim = PV_prod_sim

        # initial ES configuration
        self.observation.ES_content = 0
        self.observation.buffer = self.ES_max

    def determine_inflexible(self, day, ix):
        """
        Calculate the inflexible load based on Prosuming agent load data,PV gen data, and data index
            The data for the past hour would be the average of (ix, and ix - 1) in the dataset

        - used in updating the current observation
        """
        self.observation.inflexible_load = (self.sim_load[day][ix] + self.sim_load[day][ix-1])/2 - \
                                            (self.PV_prod_sim[day][ix] + self.PV_prod_sim[day][ix-1])/2
    
class Observations:
    def __init__(self):
        self.inflexible_load = 0.0    # average load over the past hour
        self.ES_content = 0.0         # Energy Storage battery content
        self.buffer = 0.0             # amount of space in the battery left (how much left before fully charged)

    def get_obs(self, tou, fit):
        """
        returns all observation attributes of the observation object
        """
        return [self.inflexible_load, self.ES_content, self.buffer, tou, fit]

