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
import torch.nn as nn
import time



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
    output_size = 5
    kernel_size = 5
    # middle = 196
    middle = 245
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.qnn = DQN(input_size, output_size, middle, kernel_size)
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    start = time.time()

    random_prob = 0.5
    print("{}th round and {}th step".format(game_state['round'],game_state['step']))

    if self.train and random.random() < random_prob:
         self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
         return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model)
    state_input = state_to_features(5, game_state)
    print("get state of size".format(state_input.shape))
    choice = self.qnn.choose_action(state_input)
    print("choose using QNN")
    print(choice)
    end = time.time()

    print(str(end - start))

    return ACTIONS[choice[0]]


def state_to_features(arena_len, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    self_loc = game_state['self'][-1]
    others = [xy[-1] for xy in game_state['others']]
    self_field = np.zeros((17,17))
    self_field[self_loc] = 1
    # print(self_field[others])
    for other in others:
        self_field[other] = -1
    for coin in game_state['coins']:
        self_field[coin] = 2

    game_field = game_state['field']
    explosion_field = game_state['explosion_map']

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(self_field)
    channels.append(game_field)
    channels.append(explosion_field)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels
    # return stacked_channels.reshape(-1)
