# Simulations
from matplotlib import pyplot as plt
import pandas as pd

from environment import Market, Utility, ConsumingAgent, ProsumingAgent
from maddpg import MADDPG, Critic

from torch.optim import Adam
import time
import random
import json
import copy
import pickle

import torch

"""
Data Processing:
1. Turn every row into custormer: [list of data points for that day]
2. Take an average over the whole year for each of the 300 households for each of the different categories
    - the results should be 300*3 rows for each household
3. Add the Gross Consumption (GC) and Controlled Load (CL) of each household together to get the total load, keep the Gross Generation (GG) Separate
4. Input this output into the PV_contribution_gen function and household_load_gen generator functions which yield
 the corresponding list of data for a customer 
    - PV_contribution_gen will only yield that dataset if the sum of the list is not 0
"""

with open('ausgriddata.json', 'r') as json_file:
    sim_data = json.load(json_file)

# Generating the Consumer + Prosumer Generation and Load data

def PV_contribution_gen(data):
    """
    Args:
        data: Customer dict: {'1': {"GG": [], "GC": [], "count": 0}}
            - keys --> customer name
            - values --> dictionary indicating the Gross Generation and Gross Consumption taken at 30 min increments throughout an average day
                        (also indicating the count to help compute the average)
    
    Yields:
        a list of PV energy contribution data 
        i.e. the total energy the PV contributes to the ES every 30 mins in a day from a open source data set
    """
    keys_list = list(data.keys())
    for i in range(len(keys_list)):
        customer = random.choice(keys_list)
        yield data[customer]["GG"]

def household_load_gen(data):
    """
    Yields a list of household load data:
        the total energy that a household in the dataset uses every 30 mins
    """
    keys_list = list(sim_data.keys())
    for i in range(len(keys_list)):
        customer = random.choice(keys_list)
        yield sim_data[customer]["GC"]
    # for customer in sim_data.keys():
    #     yield sim_data[customer]["GC"]

# Functions used within Training: Market Clearing, Updating Energy Balance according to load + gen

def market_clearing(buy_orders, sell_orders, market, hour):
    """
    Market Clearing Algorithm: defined by https://www.ijcai.org/proceedings/2021/0401.pdf paper
      buy_orders: buy order book k_b(x, p_x, q_x) = [[x, p_x, q_x], ...]
        - lists trading price and amt of energy requested for every buyer x
      sell_orders: sell order book k_s(y, p_y, q_y)
        - lists trading price and amt of energy to be sold for every seller y
    """
    # Market Clearing helper functions
    def sort_orders(buy, sell):
        buy = sorted(buy, key=lambda x: x[1], reverse=True) # sort buy orders from high to low price
        sell = sorted(sell, key=lambda x: x[1], reverse=False) # sort sell orders from low to high price
        return (buy, sell)

    def tensor_to_list(tensor_list):
        new_list = []
        for item in tensor_list:
            if isinstance(item, torch.Tensor):
                new_list.append(item.item())
            else:
                new_list.append(item)
        return new_list

    def remove_tensors(order):
        new_order = []
        for bid in order:
            bid = tensor_to_list(bid)
            new_order.append(bid)
        return new_order

    def dict_to_list(d):
        new_list = []
        for key, item in d.items():
            if item[1] != 0:
                new_list.append([key, item[0], item[1]])
        return new_list
    
    i, j = 0, 0 # initialize indices
    buy_orders = dict_to_list(buy_orders)
    buy_orders = remove_tensors(buy_orders)
    sell_orders = dict_to_list(sell_orders)
    sell_orders = remove_tensors(sell_orders)
    buy_orders, sell_orders = sort_orders(buy=buy_orders, sell=sell_orders)
    if (buy_orders and sell_orders):
        b_price = buy_orders[0][1]
        s_price = buy_orders[0][1]
        while b_price >= s_price:
            if (i >= len(buy_orders) or j >= len(sell_orders)):
                break

            buyer, b_price, b_quantity = buy_orders[i]
            seller, s_price, s_quantity = sell_orders[j]
            trade_quantity = min(b_quantity, s_quantity)
           
            trade_price = round((b_price + s_price)/2, 3)
            trade_money = trade_quantity*trade_price
            # update order book
            buy_orders[i][2] = buy_orders[i][2] - trade_quantity # buyer quantity still needed
            sell_orders[j][2] = sell_orders[j][2] - trade_quantity # seller quantity still can be sold
            
            # update ES Content and Reward
            if buyer != seller:
                buyer_agent, seller_agent = market.get_agent(buyer), market.get_agent(seller)
                buyer_agent.add_energy_bal(trade_quantity)
                buyer_agent.reward += trade_money # reward = cost (IF ISSUES CHANGE LATER TO ACTUAL REWARD)
                seller_agent.add_energy_bal(-trade_quantity)
                seller_agent.reward -= trade_money

            # move to the next buyer if buyer i's quantity is met
            buy_quantity_i = buy_orders[i][2]
            if buy_quantity_i == 0:
                i += 1
            sell_quantity_j = sell_orders[j][2]
            if sell_quantity_j == 0:
                j += 1
            
    # determine leftovers by seeing which consumers / prosumers have non-zero quantities
    def determine_leftovers(list_of_lists):
        leftover_list = []
        for name, price, quantity in list_of_lists:
            if quantity != 0:
                leftover_list.append([name, price, quantity])
        return leftover_list
    
    def clear_leftovers(buy_leftovers, sell_leftovers):
        """
        Clearing the leftovers with the utlilty prices
            - buy leftovers are bought from utility at Time of Use (ToU)
            - sell leftovers are sold to utlity at Feed in Tarrif (FiT)
        """
        for name, price, quantity in buy_leftovers:
            trade_price = market.get_ToU(hour)
            total_price = quantity*trade_price

            buyer_agent = market.get_agent(name)
            buyer_agent.add_energy_bal(quantity)
            buyer_agent.reward += total_price
        for name, price, quantity in sell_leftovers:
            trade_price = market.get_FiT(hour)
            total_price = quantity*trade_price

        seller_agent = market.get_agent(name)
        seller_agent.add_energy_bal(-quantity)
        seller_agent.reward += -total_price
    
    buy_leftovers = determine_leftovers(buy_orders)
    sell_leftovers = determine_leftovers(sell_orders)
    
    clear_leftovers(buy_leftovers, sell_leftovers)

def update_energy_bal_PV_load(market, day, hh):
    """
    Updates the Energy Balance each 30 minutes from simulation PV generation and load data for the whole market

    Energy Balance keeps track of the net energy for an agent between market openings
        - energy bal = energy bought + energy produced - energy sold - energy consumed
        - the net negative amount is cleared before each market opening (converted to an energy bill)
    
    Prosumers' energy balance is represented directly by their Energy Storage
    Consumers' energy balance is represented by "self.energy balance" to keep track of incurred load
    """
    for agent in market.agents:
        if agent.producing:
            agent.add_energy_bal(agent.PV_prod_sim[day][hh])             # add the energy produced directly from agent's observation of ES content
        agent.add_energy_bal(-agent.sim_load[day][hh])                   # subtract the energy consumed

def trading_alg(market, maddpg, day, hh, secondtime=False):
    """
    Functioned called at every market opening for each agent
        - calculates reward for an agent
        - adds experience to replay memory buffer
        - determines agent's action (price and energy decision) from policy 
        - appends the agent's action to the buyer's or seller's order book
    """
    # Helper function
    def add_experience(agent, orderbook, past_obs, action, obs, reward, total):
        """
        Add an experience containing: (observation, action, new observation, reward) to an
            Agent's replay memory buffer

        Used within each market opening
        """
        def orderbook_without_agent(agent, ob):
            """
            NEED TO CHANGE THE REPRESENTATION OF THE ORDERBOOK:
                - from dict --> list
                - omit names
                - also omit the agent
            """
            new_buy = []
            new_sell = []
            buy_orders, sell_orders = ob
            # print("length of sell orders", len(sell_orders))
            # print(sell_orders)
            for name, decision in buy_orders.items():
                if agent.name != name:
                    new_buy.append(decision)
            for name, decision in sell_orders.items():
                if agent.name != name:
                    new_sell.append(decision)
            if (len(new_buy) > 0 and len(new_sell) > 0):  # if both are not empty lists
                return [new_buy, new_sell]
            elif len(new_sell) > 0:
                return [new_sell,]
            elif len(new_buy) > 0:
                return [new_buy,]
            else:
                return None
        
        new_orderbook = orderbook_without_agent(agent, orderbook)
        agent.memory.push(new_orderbook, past_obs, action, obs, reward, total)    # add to experience relay with new obs state

    def update_reward_energybal(agent):
        """
        before the new orderbooks are put together,
          clear the transient energy balance (for consumers) from the previous market period
          if energy balance < 0:
            clear with utility prices
          if energy balance > 0:
            reset to 0
        """
        energy_bal = agent.get_energy_bal() # taken directly from ES if prosumer 
        if isinstance(energy_bal, torch.Tensor):
            energy_bal = energy_bal.item()
        if energy_bal < 0:
            energy_bill = -agent.get_energy_bal() * market.get_ToU(hour) # positive value
            if isinstance(energy_bill, torch.Tensor):
                energy_bill = energy_bill.item()
            agent.reward += energy_bill   # energy bill for the past hour
            agent.reset_energy_bal()                                                # reset ES content after bill has been accounted for
        if not agent.producing:
            agent.reset_energy_bal()    # still need to reset energy bal of consumers even if positive

    def add_action_to_orderbook(agent, buy_orders, sell_orders):
        """
        Adds the action from an agent to the current orderbook put together by the trading alg
        """
        agent.action = maddpg.select_action(agent, tou, fit)  # selects action based on policy or random
        if agent.producing: # PROSUMER
            buy_price, buy_quantity, sell_price, sell_quantity = agent.action
            buy_price *= tou
            buy_quantity *= (agent.ES_max - agent.get_energy_bal())           # max amount of energy they can add
            sell_price *= fit
            sell_quantity *= max(agent.get_energy_bal(), 0)
            sell_orders[agent.name] = [sell_price, sell_quantity]
            buy_orders[agent.name] = [buy_price, buy_quantity]
        else: # CONSUMER
            buy_price, buy_quantity = agent.action
            buy_price = buy_price * tou
            buy_quantity *= 4           # limiting the max to 4 KWh
            buy_orders[agent.name] = [buy_price, buy_quantity]

    hour = hh // 2
    buy_orders, sell_orders = market.init_orders()
    market.past_total = market.total_cost
    market.total_cost = 0
    tou, fit = market.get_ToU(hour), market.get_FiT(hour)
    for agent in market.agents:
        # calculate reward for every agent before we enter the market
        update_reward_energybal(agent)
        market.total_cost += agent.reward
        # add to memory experience relay
        if secondtime:
            # set the inflexible load in the observation from data
            agent.determine_inflexible(day, hh)
            obs = torch.tensor(agent.observation.get_obs(tou, fit), device=market.device)
            add_experience(agent, market.orderbook, agent.past_obs, agent.action, obs, 
                            torch.tensor([agent.reward], device=market.device), 
                            torch.tensor([market.past_total], device=market.device))
        else:
            obs = torch.tensor(agent.observation.get_obs(tou, fit), device=market.device)   # no past obs
        agent.reward = 0    # reset reward
        agent.past_obs = obs  # the obs becomes past obs
        add_action_to_orderbook(agent, buy_orders, sell_orders)
    return (sell_orders, buy_orders)

def train(market, hyperparams):
    """
    How to do the training:
        1. Select a consumer for each agent from the AUSgrid dataset
        2. Compile a PyTorch dataset for each agent
        3. Train the agents with data from every day of the year

    Args:
        market: the current market
        maddpg: the algorithm object, containing parameter update methods
    """
    
    batchsize, lr, gamma, eps_decay, tau = hyperparams
    maddpg = MADDPG(market, batchsize, lr, gamma, eps_decay, tau)
    # CLEAR ALL ENERGY BALANCES
    market.clear_all_energy()
    # CLEAR ALL MONEY
    market.clear_all_money()
    for day in range(365):
        print(f'day: {day}')
        for hh in range(48):
            update_energy_bal_PV_load(market, day, hh)
            # Market transaction and training network with randomly sampled experiences every hour
            hour = (hh) // 2
            if ((hh) % 2 == 0): # every hour
                if hour > 1:
                    # calc rewards, adds experience to memory buffer; calculates next best action, stores in outputted order books
                    (sell_orders, buy_orders) = trading_alg(market, maddpg, day, hh, secondtime=True)
                else:
                    (sell_orders, buy_orders) = trading_alg(market, maddpg, day, hh, secondtime=False)
                maddpg.episodes += 1
                market.orderbook = [copy.deepcopy(buy_orders), copy.deepcopy(sell_orders)]
                market_clearing(buy_orders, sell_orders, market, hour) # market clearing alg: updates the immediate reward, updates the energy_bal of the agents after market transaction
                # update critic and actor networks of all agents
                maddpg.update_parameters()
        market.daily_totalcosts.append(market.total_cost)

def setup_market(pros_cons, utility, sim_data):
    name = 0
    pros, cons = pros_cons
    market = Market(utility=utility, pros_cons=pros_cons, agents=[])
    for _ in range(pros):
        market.add_agent(ProsumingAgent(name, 10, 10*next(PV_contribution_gen(sim_data)), next(household_load_gen(sim_data))))
        # market.add_agent(ProsumingAgent(name, 2, 10, dummy_gen, next(household_load_gen(sim_data))))
        name += 1
    for _ in range(cons):
        market.add_agent(ConsumingAgent(name, next(household_load_gen(sim_data))))
        name += 1
    return market

# FUNCTIONS TO SAVE TRAINING DATA
def save_criticloss(market, folder):
    all_criticloss = []
    for agent in market.agents:
        critic_loss = [loss.item() for loss in agent.critic_loss]
        all_criticloss.append(critic_loss)
        plt.plot(critic_loss, label=agent.name)
    plt.title("Critic Loss")
    plt.legend()
    plt.savefig(folder + "/criticloss.png")
    plt.close()

    with open(folder + "/criticloss.pkl", "wb") as file:
        pickle.dump(all_criticloss, file)

def save_actorloss(market, folder):
    all_actorloss = []
    for agent in market.agents:
        actor_loss = [loss.item() for loss in agent.actor_loss]
        all_actorloss.append(actor_loss)
        plt.plot(actor_loss, label=agent.name)
    plt.title("Actor Loss")
    plt.legend()
    plt.savefig(folder + "/actorloss.png")
    plt.close()

    with open(folder + "/actorloss.pkl", "wb") as file:
        pickle.dump(all_actorloss, file)

def save_energybill(market, folder):
    """
    Saves the amount of money spent on all energy expenses
      for a whole day during training
    """
    plt.plot(market.daily_totalcosts)
    plt.title("Total Energy Bills")
    plt.xlabel("days of training")
    plt.savefig(folder + "/energybill.png")
    plt.close()

    with open(folder + "/energybill.pkl", "wb") as file:
        pickle.dump(market.daily_totalcosts, file)

def save_marketobject(market, folder):
    with open(folder + "/market.pkl", "wb") as file:
        pickle.dump(market, file)