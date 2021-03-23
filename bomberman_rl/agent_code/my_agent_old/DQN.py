import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 32
LR = 0.01                   # learn rate
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100   # how often do the reality network iterate
MEMORY_CAPACITY = 2000

N_ACTIONS = 6  # action space
N_STATES = 1

class DQN(object):
	def __init__(self,in_channels:int,out_channels:int,middle:int,kernel_size:int,action_space = 6,stride = 2):

		# eval for the evaluation network, target is the reality network
		self.eval_net = Net(in_channels,out_channels,middle,kernel_size,action_space ,stride )
		self.target_net = Net(in_channels,out_channels,middle, kernel_size,action_space ,stride )

		# how often do the reality network iterate
		self.learn_step_counter = 0

		self.memory_counter = 0
		self.memory = list(np.zeros((MEMORY_CAPACITY, N_ACTIONS)))  # init memory
		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
		self.loss_func = nn.MSELoss()

	def choose_action(self, x):
		"""
		x: input, state of the agent
		"""
		x = torch.unsqueeze(torch.FloatTensor(x), 0)
		print(x.shape)

		if np.random.uniform() < EPSILON:  # choose an optimize function
			actions_value = self.eval_net.forward(x)  # put the state in nn
			# torch.max(input,dim)返回dim最大值并且在第二个位置返回位置比如(tensor([0.6507]), tensor([2]))
			action = torch.max(actions_value, 1)[1].data.numpy()  # 返回动作最大值
		else:
			# choose an action randomly
			action = np.array([np.random.randint(0, N_ACTIONS)])
		return action

	def store_transition(self, s, a, r, s_):
		# if the memory is full, then we cover the old memory
		index = self.memory_counter % MEMORY_CAPACITY
		self.memory[index] = [s, a, r, s_]
		self.memory_counter += 1

	def learn(self):
		# target net iteration
		if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
			# copy all parameter from eval_net to target_net
			self.target_net.load_state_dict(self.eval_net.state_dict())
		self.learn_step_counter += 1
		# get batch from memory, find the labels of the batch size
		sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
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
		q_target = b_r + GAMMA * q_next.max(1)[0]  # shape (batch, 1) DQL核心公式
		loss = self.loss_func(q_eval, q_target)  # 计算误差
		# 计算, 更新 eval net
		self.optimizer.zero_grad()  #
		loss.backward()  # 反向传递
		self.optimizer.step()


class Net(nn.Module):
	def __init__ (self,in_channels:int,out_channels:int,middle:int,kernel_size:int,action_space = 6,stride = 2,dropout_f = 0.01):
		super(Net, self).__init__()
		self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0)
		self.f1 = nn.Linear(middle, 12)
		self.f1.weight.data.normal_(0, 0.1)
		self.f2 = nn.Linear(12, action_space)
		self.f2.weight.data.normal_(0, 0.1)
        # self.drop = nn.Dropout(dropout_f)

	def forward(self, x):
		x = self.c1(x)
		x = F.relu(x)
		x = x.view(x.size(0), -1)
		x = self.f1(x)
		x = F.relu(x)
		action = self.f2(x)
		return action

