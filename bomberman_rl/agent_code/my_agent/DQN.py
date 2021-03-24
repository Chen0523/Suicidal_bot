import numpy as np
import random
import math
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 32
LR = 0.01                   # learn rate
EPSILON_L = 0.5
EPSILON_H = 0.9
GAMMA = 0.8
TARGET_REPLACE_ITER = 200   # how often do the reality network iterate

N_ACTIONS = 6  # action space
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class DQN(object):
	def __init__(self,in_channels:int,out_channels:int,kernel_size:int,MEMORY_CAPACITY,MIN_ENEMY_STEPS,action_space = 6,stride = 2, ):

		# eval for the evaluation network, target is the reality network
		self.eval_net = Net(in_channels,out_channels,kernel_size,action_space ,stride )
		self.target_net = Net(in_channels,out_channels, kernel_size,action_space ,stride )

		# how often do the reality network iterate
		self.learn_step_counter = 0

		self.memory_counter = 0
		self.memory = list(np.zeros((MEMORY_CAPACITY, N_ACTIONS)))  # init memory
		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = LR)
		self.loss_func = nn.MSELoss()
		self.loss = 0
		self.MEMORY_CAPACITY = MEMORY_CAPACITY
		self.MIN_ENEMY_STEPS = MIN_ENEMY_STEPS

	def choose_action(self, x,train = False):
		"""
		x: input, state of the agent
		"""
		x = torch.unsqueeze(torch.FloatTensor(x), 0)
		EPSILON = EPSILON_H
		# print(x.shape)
		if train:

			if self.memory_counter < self.MEMORY_CAPACITY:
				# now only enemy steps are recorded, my_agent just hanging around
				useless_action = [np.random.choice([0, 1, 2, 3, 4, 5], p=[.2, .2, .2, .2, .1, .1])]
				return useless_action
			elif self.memory_counter < self.MEMORY_CAPACITY * 2:
				EPSILON = EPSILON_L

			if self.memory_counter == self.MEMORY_CAPACITY:
				print("Stop making random choices")

		# Start doing real action
		if np.random.uniform() < EPSILON:  # choose an optimize function
			actions_value = self.eval_net.forward(x)  # put the state in nn
			# torch.max(input,dim)返回dim最大值并且在第二个位置返回位置比如(tensor([0.6507]), tensor([2]))
			action = torch.max(actions_value, 1)[1].data.numpy()  # choose the action that returns max. reward
		else:
			# choose an action randomly
			action = [np.random.choice([0,1,2,3,4,5], p=[.2, .2, .2, .2, .1, .1])]
			# action = np.array([np.random.randint(0, N_ACTIONS)])
		return action

	def store_transition(self, s, a, r, s_):
		# if the memory is full, then we cover the old memory
		index = self.memory_counter % self.MEMORY_CAPACITY
		self.memory[index] = [s, a, r, s_]
		self.memory_counter += 1

	def learn(self):
		# target net iteration
		if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
			# copy all parameter from eval_net to target_net
			print("iterating target net")
			self.target_net.load_state_dict(self.eval_net.state_dict())
		self.learn_step_counter += 1
		# get batch from memory, find the labels of the batch size
		sample_index = np.random.choice(self.MEMORY_CAPACITY, BATCH_SIZE)
		b_s = []
		b_a = []
		b_r = []
		b_s_ = []
		for i in sample_index:
			b_s.append(self.memory[i][0])
			b_a.append(np.array(self.memory[i][1], dtype=np.int32))
			b_r.append(np.array([self.memory[i][2]], dtype=np.int32))
			b_s_.append(self.memory[i][3])

		b_s = torch.FloatTensor(b_s)  # 取出s
		b_a = torch.LongTensor(b_a)  # 取出a
		b_r = torch.FloatTensor(b_r)  # 取出r
		b_s_ = torch.FloatTensor(b_s_)  # 取出s_

		# 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
		# choose the behavior according to b_a
		q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) 找到action的Q估计(关于gather使用下面有介绍)
		q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach Q现实
		q_next_max =  q_next.max(1)[0]
		q_target = b_r + GAMMA * torch.unsqueeze(q_next_max,1)  # shape (batch, 1) DQL核心公式
		self.loss = self.loss_func(q_eval, q_target)  # 计算误差
		# 计算, 更新 eval net
		self.optimizer.zero_grad()  #
		self.loss.backward()  # 反向传递
		self.optimizer.step()


class Net(nn.Module):
	def __init__ (self,in_channels:int,out_channels:int,kernel_size:int,action_space = 6,stride = 2,dropout_f = 0.01):
		super(Net, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, int(kernel_size / 2), int(stride/2), 0),
			nn.ReLU(),
		)
		conv_out_size = self._get_conv_out()

		self.fc = nn.Sequential(
			nn.Linear(conv_out_size, 12),
			nn.ReLU(),
			nn.Linear(12, action_space)
		)

	def _get_conv_out(self):
		o = self.conv(torch.zeros(1,3,13,13))
		return int(np.prod(o.size()))

	def forward(self, x):
		conv_out = self.conv(x).view(x.size()[0], -1)
		return self.fc(conv_out)


