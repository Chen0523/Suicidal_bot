import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

import pickle
import random
from DQN import DQN
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import time

# python main.py play --my-agent my_agent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    input_size = 3
    output_size = 6
    kernel_size = 4
    self.total_rewards = 0
    self.record = pd.DataFrame(columns=["round", "steps", "loss", "total_rewards"])

    # training parameters
    self.MIN_ENEMY_STEPS = 15000
    self.MEMORY_CAPACITY = 15000

    self.modelpath = os.path.join(os.getcwd(),"models",'model.pt')
    self.qnn = DQN(input_size, output_size, kernel_size, self.MEMORY_CAPACITY, self.MIN_ENEMY_STEPS)

    # if self.train or not os.path.isfile(self.modelpath):
    if self.train:
        print("no model, setting up from scratch")
        self.logger.info("Setting up model from scratch.")

    else:
        self.logger.info("Loading model from saved state.")

        state_dict = torch.load((self.modelpath), map_location=lambda storage, loc: storage)
        self.qnn.eval_net.load_state_dict(state_dict)
        print("loaded model from saved state")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    state_input = state_to_features(game_state)

    if self.train:
        self.logger.debug("Traning action")
        choice = self.qnn.choose_action(state_input,True)
    else:
        self.logger.debug("Querying model for action.")
        choice = self.qnn.choose_action(state_input)

    return ACTIONS[choice[0]]


def state_to_features( game_state: dict, is_enemy = False) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends

    if game_state is None:
        #initial statues
        self_field = np.zeros((17, 17))
        game_field = np.zeros((17, 17))
        explosion_field = np.zeros((17, 17))
    else:
        #location map
        self_loc = game_state['self'][-1]

        others = [xy[-1] for xy in game_state['others'] if xy[0]!= 'my_agent']
        self_field = np.zeros((17,17))
        self_field[self_loc] = 1

        for other in others:
            self_field[other] = -1
        for coin in game_state['coins']:
            self_field[coin] = 2

        #game map
        game_field = game_state['field']

        #explosion map
        explosion_field = game_state['explosion_map']

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(self_field)
    channels.append(game_field)
    channels.append(explosion_field)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)

    return stacked_channels

