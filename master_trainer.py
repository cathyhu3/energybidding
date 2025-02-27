import os
import json
import numpy as np

from environment import *
from train import *
from simulation import *
from maddpg import *

"""
Hyperparameter tuning, saving all results
- trains the model with many different hyperparameters
- simulates the outcome (tests) of all of these models
"""

def training(market, hyperparams, folder):
    """
    Trains the models within the market given the 
        hyperparameters using functions within the training module
    Saves the Critic and Actor loss into the specified folder
    """
    train(market, hyperparams) # make train change the hyperparameters within maddpg object
    save_actorloss(market, folder)
    save_criticloss(market, folder)
    save_energybill(market, folder)
    save_marketobject(market, folder)

def simulation(market, folder, untrained=False, day=None):
    """
    Runs all simulations to test the performance of the
        (trained or untrained) model using functions within simulation module
    Saves simulation data into specified folder
    """
    day = simulate_day(market, day, random_action=untrained)
    # save_marketcosts(market, folder)
    save_all(market, folder, day)
    if not untrained:
        save_energyplot(market, folder)
        save_moneyplot(market, folder)
        save_sim_video(market, folder)
    return day

def mutliday_sim(market, num_days, folder):
    days = []
    for i in range(num_days):
        if i == 0:
            day = simulate_day(market, day=None, random_action=False, first_time=True)
        else:
            day = simulate_day(market, day=None, random_action=False, first_time=False)
        days.append(day)
    save_all(market, folder, days)
    save_energyplot(market, folder)
    save_moneyplot(market, folder)
    save_sim_video(market, folder)
    

# load simulation data
with open('ausgriddata.json', 'r') as json_file:
    sim_data = json.load(json_file)

# establish num of prosumers and consumers + utility
pros_cons = [3,2]
ToU = [0.08 for i in range(8)] + [0.13 for i in range(8)] + [0.18 for i in range(4)] + [0.08 for i in range(4)]
FiT = [0.04 for i in range(24)]
utility = Utility(ToU, FiT)

def runit(pros_cons, utility, sim_data):
    """
    Iterates through a range of hyperparameters:
        trains the model (the respective NNs within each agent of the market)
        simualtes the model
        saves trained model and results in a specified folder on computer
    """
    sim_num = 0
    for batchsize in [32,]:
        for lr in [0.001,]:
            for gamma in [0.95,]:
                for eps_decay in [700,]:
                    for tau in [0.02,]:
                        for epoch in range(1):
                            print("SIM NUM", sim_num)
                            market = setup_market(pros_cons, utility, sim_data)
                            parent = f"/Users/cathyhu/Downloads/ORNL ECO/decentralized/training0830/sim{sim_num}_epoch{epoch}_bsize{batchsize}_lr{lr}_g{gamma}_edr{eps_decay}_t{tau}"
                            trained_folder = "trained"
                            untrained_folder = "untrained"
                            # Create the parent folder
                            if not os.path.exists(parent):
                                os.makedirs(parent)
                            # Create the subfolders within the parent folder
                            trained_path = os.path.join(parent, trained_folder)
                            untrained_path = os.path.join(parent, untrained_folder)
                            os.makedirs(trained_path, exist_ok=True)
                            os.makedirs(untrained_path, exist_ok=True)

                            hyperparams = [batchsize, lr, gamma, eps_decay, tau]
                            training(market, hyperparams, folder=parent)
                            mutliday_sim(market, num_days=3, folder=trained_path)
                            
                            # single-day simulation instead of multi-day simulation
                            # day = simulation(market, folder=trained_path, untrained=False)
                            # simulation(market, folder=untrained_path, untrained=True, day=day)

                        sim_num += 1

runit(pros_cons, utility, sim_data)
