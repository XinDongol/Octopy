import torch
from torch import nn, autograd

import numpy as np
import random
from torch.autograd import Variable
from torch.utils import data
from warehouse.funcs import save_checkpoint
import torch.nn.functional as F

def update_list(old_l, new_l):
	'''
	Update the old list (old_l) in-place with new list (new_l)
	'''
	[old_l.pop() for i in range(len(old_l))]
	old_l.extend(new_l)

class User(object):
	def __init__(self, user_index, ready_model, all_users_opt, train_loader, test_loader):
		#print('U-1')
		# self.update_local_config(config.users[user_index])
		self.user_opt = all_users_opt[user_index]
		self.user_index = user_index


		self.train_loader, self.test_loader = train_loader, test_loader
		update_list(self.train_loader.sampler.indices, self.user_opt['local_train_dataset'])
		update_list(self.test_loader.sampler.indices, self.user_opt['local_test_dataset'])

		self.net = ready_model
		self.device = torch.device(next(self.net.parameters()).device)
		#print('U-2: ', type(self.device))
		#print('U-3')


	# def update_local_config(self, local_config):
	# 	self.__dict__.update(local_config)

	def get_optimizer(self):
		if self.user_opt['optimizer'] == 'SGD':
			return torch.optim.SGD(self.net.parameters(), lr=self.user_opt['lr'],
									weight_decay=0)
		elif self.user_opt['optimizer'] == 'Adam':
			return torch.optim.Adam(self.net.parameters(), lr=self.user_opt['lr'], betas=(0.9,0.99),
									weight_decay=0)
		else:
			raise NotImplementedError("Please implement %s in User.get_optimizer"%self.user_opt['optimizer'])

	def get_loss_func(self):
		if self.user_opt['loss_func'] == 'CrossEntropyLoss':
			return nn.CrossEntropyLoss()
		else:
			raise NotImplementedError("Please implement %s in User.get_optimizer"%self.user_opt['loss_func'])

	def local_train(self):
		# print('Starting the training of user: ', self.user_index)
		optimizer = self.get_optimizer()
		loss_func = self.get_loss_func()
		self.net.train()
		for epoch in range(1, self.user_opt['local_epoch'] + 1):
			#print('LOL, I am training...')
			for batch_idx, (data, target) in enumerate(self.train_loader):
				#print('U-3: ', target)
				data, target = data.to(self.device), target.to(self.device)
				#print('U-4: ', data)
				optimizer.zero_grad()
				#print('U-5: ', type(self.net))
				output = self.net(data)
				#print('U-6: ')
				loss = loss_func(output, target)
				#print('U-7: ')
				loss.backward()
				#print('U-8: ')
				optimizer.step()

		# save_checkpoint(self.net, filename='checkpoint_%d.pth' % self.user_index)		
		


