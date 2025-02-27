# from train import update_energy_bal_PV_load
import os

import random
import time
import copy
import torch
import pickle
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from animated_sim import update_plot, generate_well_connected_graph, figure_setup

"""
Market Clearing outcomes:
[
    {(seller, buyer): [quantity, price, total], ...},   hour 1
    {(seller, buyer): [quantity, price, total], ...},   hour 2
    ...
]
"""
# Functions used during Simulation
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

    def add_lists(list1, list2):
        return [val1 + val2 for val1, val2 in zip(list1, list2)]

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
    market.orderbook = [copy.deepcopy(buy_orders), copy.deepcopy(sell_orders)]
    market.orderbooks.append(market.orderbook)
    if (buy_orders and sell_orders):
        b_price = buy_orders[0][1]
        s_price = buy_orders[0][1]
        while b_price >= s_price:
            if (i >= len(buy_orders) or j >= len(sell_orders)):
                break

            buyer, b_price, b_quantity = buy_orders[i]
            seller, s_price, s_quantity = sell_orders[j]

            trade_quantity = min(b_quantity, s_quantity)
            trade_price = (b_price + s_price)/2
            trade_money = trade_quantity*trade_price
            # update order book
            buy_orders[i][2] = buy_orders[i][2] - trade_quantity # buyer quantity still needed
            sell_orders[j][2] = sell_orders[j][2] - trade_quantity # seller quantity still can be sold
            
            # update ES Content and Reward
            buyer_agent, seller_agent = market.get_agent(buyer), market.get_agent(seller)
            buyer_agent.add_energy_bal(trade_quantity)
            buyer_agent.reward += trade_money # reward = cost (IF ISSUES CHANGE LATER TO ACTUAL REWARD)
            seller_agent.add_energy_bal(-trade_quantity)
            seller_agent.reward -= trade_money
            market.outcome[(seller, buyer)] = [round(trade_quantity,2), round(trade_price,3), round(trade_money,3)]

            # move to the next buyer if buyer i's quantity is met
            buy_quantity_i = buy_orders[i][2]
            if buy_quantity_i == 0:
                i += 1
            sell_quantity_j = sell_orders[j][2]
            if sell_quantity_j == 0:
                j += 1
            
    def leftovers(list_of_lists):
        """
        determine leftovers by seeing which consumers / prosumers have non-zero quantities
        """
        leftover_list = []
        for name, price, quantity in list_of_lists:
            if quantity != 0:
                leftover_list.append([name, price, quantity])
        return leftover_list
    buy_leftovers = leftovers(buy_orders)
    sell_leftovers = leftovers(sell_orders)
    tou, fit = market.get_ToU(hour), market.get_FiT(hour)
    # clear leftovers with utility prices
    for name, price, quantity in buy_leftovers:
        trade_price = tou
        total_price = quantity*trade_price

        buyer_agent = market.get_agent(name)
        buyer_agent.add_energy_bal(quantity)
        buyer_agent.reward += total_price
        # update_transactions(market.transactions, name, "u", [quantity, total_price])
        # print(f'utility sold {quantity}kWh to buyer {name} for {trade_price} per kWh (total of ${total_price})')
        market.outcome[('utility', name)] = [round(quantity,2), round(trade_price,3), round(total_price,3)]

    for name, price, quantity in sell_leftovers:
        # store extra prosumer energy into ES
        trade_price = fit
        total_price = quantity*trade_price

        seller_agent = market.get_agent(name)
        seller_agent.add_energy_bal(-quantity)
        seller_agent.reward -= total_price
        # update_transactions(market.transactions, "u", name, [quantity, total_price])
        # print(f'seller {name} sold {quantity}kWh to utility for {trade_price} per kWh (total of ${total_price})')
        market.outcome[(name, 'utility')] = [round(quantity,2), round(trade_price,3), round(total_price,3)]

def trading_alg(market, day, hh, random_action, secondtime=False):
    """
    Functioned called at every market opening for each agent
        - calculates reward for an agent
        - determines agent's action (price and energy decision) from policy 
        - appends the agent's action to the buyer's or seller's order book
    """
    def use_policy(agent, obs, batchsize=None):
        with torch.no_grad():
            return agent.actor(obs)

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
        if energy_bal < 0:
            energy_bill = -energy_bal * tou
            if isinstance(energy_bill, torch.Tensor):
                energy_bill = energy_bill.item()
            market.outcome[("utility", agent.name)] = [-round(energy_bal,3),tou,round(energy_bill,3)]
            agent.reward += energy_bill   # energy bill for the past hour
            agent.reset_energy_bal()                                                # reset ES content after bill has been accounted for
        elif not agent.producing:
            agent.reset_energy_bal()    # still need to reset energy bal of consumers even if positive

    def add_action_to_orderbook(agent, buy_orders, sell_orders):
        """
        Adds the action from an agent to the current orderbook put together by the trading alg
        """
        agent.action = use_policy(agent, obs, hour)  # selects action based on policy or random
        if agent.producing: # PROSUMER
            if random_action:
                agent.action = torch.tensor([random.random(), random.random(), random.random(), random.random()], device=market.device)
            buy_price, buy_quantity, sell_price, sell_quantity = agent.action

            buy_price *= tou
            buy_quantity *= (agent.ES_max - agent.get_energy_bal())
            sell_price *= fit
            sell_quantity *= max(agent.get_energy_bal(), 0)

            sell_orders[agent.name] = [sell_price, sell_quantity]
            buy_orders[agent.name] = [buy_price, buy_quantity]
        else: # CONSUMER
            if random_action:
                agent.action = torch.tensor([random.random(), random.random()], device=market.device)
            buy_price, buy_quantity = agent.action

            buy_price = buy_price * tou
            buy_quantity *= 4           # max is 4 KWh

            buy_orders[agent.name] = [buy_price, buy_quantity]

    hour = hh // 2
    total_money = 0
    buy_orders, sell_orders = market.init_orders()
    tou, fit = market.get_ToU(hour), market.get_FiT(hour)
    for agent in market.agents:
        # calculate reward for every agent before we enter the market
        update_reward_energybal(agent)
        # update total market cost an agent money / reward
        agent.money = agent.money - agent.reward
        total_money += agent.money
        agent.reward = 0    # reset reward
        # retrieve observation
        agent.determine_inflexible(day, hh)
        obs = torch.tensor(agent.observation.get_obs(tou, fit), device=market.device)
        # action selection and orderbook updating
        add_action_to_orderbook(agent, buy_orders, sell_orders)
    market.sim_totalcost.append(total_money)
    market.sim_totalcost.append(total_money)
    return (sell_orders, buy_orders)

# Simulation of a Day Function
def simulate_day(market, day, random_action=False, first_time=False):

    # Clear all 
    if not random_action:
        day = random.randrange(0,365) # pick a random day
    # Clear all energy balances, money, and recorded orderbooks, outcomes
    if first_time:
        market.clear_all_energy()
        market.clear_all_money()
        market.orderbooks = []
        market.outcomes = []
    # print("START SIMULATION")
    # print(f'Simulation Day {day}')
    for hh in range(48):
        # time.sleep(1)
        update_energy_bal_PV_load(market, day, hh)
        # record the energy balance and money
        for agent in market.agents:
            agent.energy_bal_record.append(agent.get_energy_bal())
            agent.money_record.append(agent.money)
        hour = (hh) // 2
        # Market opening
        if ((hh) % 2 == 0):
            if hour > 0:
                (sell_orders, buy_orders) = trading_alg(market, day, hh, random_action, secondtime=True) # adds experience to memory buffer; calculates next best action, stores in order books
            else:
                market.orderbooks.append([[],[]]) # blank orderbook to match first market outcome (when all load before market opening at hour 1 is )
                (sell_orders, buy_orders) = trading_alg(market, day, hh, random_action, secondtime=False)
            # clear market
            market.outcomes.append(market.outcome)  # save to memory
            market.outcome = {}
            market_clearing(buy_orders, sell_orders, market, hour) # market clearing alg: updates the immediate reward, updates the energy_bal of the agents after market transaction
    return day

# Saving Simulation Data Helper Functions
def save_energyplot(market, dir):
    for agent in market.agents:
        plt.plot(agent.energy_bal_record, label=agent.name)
    plt.legend(loc='upper right')
    plt.title("Energy Balance")
    plt.xlabel("Time (24 hours)")
    plt.savefig(dir + "/energy.png")
    plt.close()

def save_moneyplot(market, dir):
    for agent in market.agents:
        plt.plot(agent.money_record, label=agent.name)
    total = market.sim_totalcost[-1]
    plt.plot(market.sim_totalcost, label=f"Market (Total = {total})")
    plt.legend(loc='upper right')
    plt.title("Money")
    plt.xlabel("Time (24 hours)")
    plt.savefig(dir + "/money.png")
    plt.close()

def save_all(market, dir, days):
    plt.close()
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()

    def concated_data_for_days(agent_data, days):
        concated_data = []
        for day in days:
            concated_data.extend(agent_data[day])
        return concated_data
    
    new_agent_loads = []
    new_agent_gens = []
    for agent in market.agents:
        new_agent_loads.append(concated_data_for_days(agent.sim_load, days))
        if agent.producing:
            new_agent_gens.append(concated_data_for_days(agent.PV_prod_sim, days))
        else:
            total_len = round(len(agent.sim_load[0])*len(days))
            new_agent_gens.append([0 for i in range(total_len)])

    for i, agent in enumerate(market.agents):
        ax[0].plot(agent.money_record, label=agent.name)
        ax[1].plot(agent.energy_bal_record, label=agent.name)
        ax[2].plot(new_agent_loads[i], label=agent.name)
        ax[3].plot(new_agent_gens[i], label=agent.name)
    ax[0].plot(market.sim_totalcost, label="Market")

    for subplot in ax:
        subplot.legend(loc="upper right")
        subplot.set(xlabel='Time (half hour increments)')

    ax[0].set_title("Money")
    ax[1].set_title("Energy Balance")
    ax[2].set_title("Load Profiles")
    ax[3].set_title("Generation Profiles")

    plt.tight_layout()
    plt.savefig(dir + "/all_profiles.png") 
    plt.close()

def save_sim_video(market, dir):
    pros_cons = market.num_pros_cons

    money_records = [agent.money_record for agent in market.agents]
    with open(dir + "/moneyrecords.pkl", "wb") as file:
        pickle.dump(money_records, file)
    energy_records = [agent.energy_bal_record for agent in market.agents]
    with open(dir + "/energybals.pkl", "wb") as file:
        pickle.dump(energy_records, file)

    fig, pos, ax, variables, lines, table, outbox = figure_setup(pros_cons, money_records, energy_records)

    # Animate
    ani = animation.FuncAnimation(
        fig, update_plot, fargs=(lines, variables, ax, outbox, market.orderbooks, market.outcomes),
        frames=48*3, interval=2000, blit=True
    )

    # Save the animation as a video
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(dir + '/sim_video.mp4', writer=writer)
    plt.close(fig)

    with open(dir + "/orderbooks.pkl", "wb") as file:
        pickle.dump(market.orderbooks, file)
    with open(dir + "/outcomes.pkl", "wb") as file:
        pickle.dump(market.outcomes, file)

