import os
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.colors import n_colors
from statistics import mean

from simulation import *

"""
ADD THE AVERAGE GENERATION COST OF THE WHOLE MARKET
"""

"""
10 1-day simulations on a market object

1. Run 10 simulations for each day

2. Collect 

    generation_cost data = 
        {
            1: [day 1 total, day 2 total, ..., day 40 total],
            ...
            n: [day 1 total, day 2 total, ..., day 40 total],
            market: [day 1 total, day 2 total, ..., day 40 total],
        }

    average_daily_load =            # add the lists together, then in the end divide by the total num of market openings
        {
            1: [average daily load],
            2: [average daily load],
            ...
        }

    average_daily_gen =             # add the lists together, then in the end divide by the total num of market openings
        {
            1: [average daily load],
            2: [average daily load],
            ...
        }

3. Plot

    - plot generation_cost data on plotly ridgeline plot, and also on a box plot
    - plot average_daily_load, average_daily gen on combined matplotlib graph

"""
folder0822 = "/Users/cathyhu/Downloads/ORNL ECO/decentralized/training0822/sim8_epoch2_bsize32_lr0.001_g0.95_edr700_t0.01"
folder0826 = "/Users/cathyhu/Downloads/ORNL ECO/decentralized/training0826/sim0_epoch0_bsize32_lr0.001_g0.95_edr700_t0.02"
folder0827 = "/Users/cathyhu/Downloads/ORNL ECO/decentralized/training0827/sim5_epoch2_bsize32_lr0.001_g0.9_edr700_t0.03"
folder0828 = "/Users/cathyhu/Downloads/ORNL ECO/decentralized/training0828/sim1_epoch2_bsize32_lr0.001_g0.9_edr700_t0.02"


num_days = 10

with open(folder0822 + "/market.pkl", "rb") as file:
    market = pickle.load(file)

num_agents = len(market.agents)
gen_cost_data = {}
avg_daily_gen_data = {agent.name: [0 for i in range(48)] for agent in market.agents}
avg_daily_load_data = {agent.name: [0 for i in range(48)] for agent in market.agents}

def run_and_collect(market):
    def add_lists(lista, listb):
        return [a+b for a,b in zip(lista, listb)]

    def divide_dictlist_by(dicta, x):
        for key, list_val in dicta.items():
            dicta[key] = [a/x for a in list_val]

    for _ in range(num_days):
        day = simulate_day(market, day=None, random_action=False, first_time=True)
        for agent in market.agents:
            total_cost = agent.money_record[-1]
            if agent.producing:
                avg_daily_gen = agent.PV_prod_sim[day]
            else:
                avg_daily_gen = [0 for i in range(48)]
            avg_daily_load = agent.sim_load[day]

            gen_cost_data.setdefault(agent.name, []).append(total_cost)
            avg_daily_gen_data[agent.name] = add_lists(avg_daily_gen_data[agent.name], avg_daily_gen)
            avg_daily_load_data[agent.name] = add_lists(avg_daily_load_data[agent.name], avg_daily_load)
        gen_cost_data.setdefault("market", []).append(market.sim_totalcost[-1])

    divide_dictlist_by(avg_daily_gen_data, num_days)
    divide_dictlist_by(avg_daily_load_data, num_days)

# HELPERS FOR ridge_box_figure
def ridgeplot_gen_cost(folder):
    fig = go.Figure()
    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', num_agents, colortype='rgb')
    for name, data_line, color in zip(gen_cost_data.keys(), gen_cost_data.values(), colors):
        # fig.add_trace(go.Violin(x=data_line, line_color=color, name=name))
        fig.add_trace(go.Violin(y=data_line, line_color=color, name=name, box_visible=True, meanline_visible=True))
    # fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_layout(xaxis_showgrid=False)
    fig.update_layout(
        title="Daily Energy Cost",
        xaxis_title="agent name",
        yaxis_title="energy cost"
    )
    fig.write_image(folder + "violin_energy_cost.png")

def box_gen_cost(folder):
    fig, ax = plt.subplots()
    names = gen_cost_data.keys()
    ax.boxplot(gen_cost_data.values())
    ax.set_xticklabels(names)
    ax.set_title("Daily Energy Cost")
    ax.set_xlabel("agent name")
    ax.set_ylabel("energy cost")
    plt.savefig(folder + "box_energy_cost.png")

# def ridge_box_figure(folder):

#     fig, ax = plt.subplots(1, 2, figsize=(12,5))
#     box_gen_cost(ax[0])
#     ridgeplot_gen_cost(ax[1])
#     plt.title("Daily Generation Cost")
#     plt.savefig(folder + "generation_costs.png")

def avg_load_gen_plot(folder):
    fig, ax = plt.subplots(2, 1, figsize=(5,10))
    def plot_load(load_ax):
        for name, load in avg_daily_load_data.items():
            load_ax.plot(load, label=name)
        load_ax.set_title("Average Daily Load")
        load_ax.legend(loc='upper right')
    def plot_gen(gen_ax):
        for name, gen in avg_daily_gen_data.items():
            gen_ax.plot(gen, label=name)
        gen_ax.set_title("Average Daily Generation")
        gen_ax.legend(loc='upper right')
    
    plot_load(ax[0])
    plot_gen(ax[1])
    plt.savefig(folder + "load_gen.png")

sim_analyze_folder = "/Users/cathyhu/Downloads/ORNL ECO/sim analysis/0822/"

if not os.path.exists(sim_analyze_folder):
    os.makedirs(sim_analyze_folder)

run_and_collect(market)
ridgeplot_gen_cost(sim_analyze_folder)
box_gen_cost(sim_analyze_folder)
avg_load_gen_plot(sim_analyze_folder)




# import plotly.graph_objects as go
# from plotly.colors import n_colors
# import numpy as np

# # 12 sets of normal distributed random data, with increasing mean and standard deviation
# data = (np.linspace(1, 2, 12)[:, np.newaxis] * np.random.randn(12, 200) +
#             (np.arange(12) + 2 * np.random.random(12))[:, np.newaxis])

# colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb')

# fig = go.Figure()
# for data_line, color in zip(data, colors):
#     fig.add_trace(go.Violin(x=data_line, line_color=color))

# fig.update_traces(orientation='h', side='positive', width=3, points=False)
# fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
# fig.show()

