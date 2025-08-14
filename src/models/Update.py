import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import random
import copy
from models.test import test_img_poison
from sklearn import metrics
from copy import deepcopy
import math
from torch.nn import CrossEntropyLoss



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # image: torch.Size([1, 28, 28]), torch.float32; label: int
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters())
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdatePoison(object):
    # used for MNIST experiments
    def __init__(self, args, dataset, idxs, user_idx):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        self.dataset = dataset
        
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.user_idx = user_idx

        self.poison_attack = False

        self.attacker_flag = False
    def train(self, net):
        net.train()
        # train and update
        shared_weights = copy.deepcopy(net.state_dict())
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        optimizer = torch.optim.Adam(net.parameters())
        

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.args.attack_mode == "poison" and self.args.poison_ratio > 0:
                    
                    # 批量投毒
                    total_samples = len(labels)
                    poison_num = int(total_samples * self.args.poison_ratio)
                    poison_indices = np.random.choice(total_samples, poison_num, replace=False)
                    
                    for label_idx in poison_indices:
                    
                            self.poison_attack = True
                            self.attacker_flag = True
                            original_label = labels[label_idx].item()
                            
                            
                            possible_labels = list(range(10))  
                            possible_labels.remove(original_label)
                            labels[label_idx] = torch.tensor(random.choice(possible_labels), dtype=torch.long)
                        
                        # if labels[label_idx] == 1:
                        #     self.poison_attack = True
                        #     self.attacker_flag = True
                        #     labels[label_idx] = 0
                
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # model replacement
        trained_weights = copy.deepcopy(net.state_dict())
        scale_up = 10
        # if self.args.attack_mode == "poison" and self.poison_attack:
            # print( "Scale up: {} for non-iid MNIST training".format(scale_up) )
            # print( "Poisoning test acc", test_img_poison(copy.deepcopy(net), self.dataset, self.args) )
            # print("Poisoning weights...")
        attack_weights = copy.deepcopy(shared_weights)
        for key in shared_weights.keys():
            difference =  trained_weights[key] - shared_weights[key]
            attack_weights[key] += scale_up * difference

        return attack_weights, sum(epoch_loss) / len(epoch_loss), self.attacker_flag

        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.attacker_flag


class LocalUpdateBack(object):
    # used for MNIST experiments
    def __init__(self, args, dataset, idxs, user_idx, trigger=np.ones([3, 3]), start_x=0, start_y=0):
        self.args = args
        self.loss_func = CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.user_idx = user_idx
        self.poison_attack = False
        self.attacker_flag = False
        self.trigger = trigger
        self.start_x = start_x
        self.start_y = start_y
        self.from_label = None 
        self.to_label = 0  
    
    def replace_data(self, image, trigger, start_x, start_y):

        h, w = trigger.shape

        if isinstance(trigger, np.ndarray):
            trigger = torch.tensor(trigger, dtype=image.dtype, device=image.device)

        if image.dim() == 3:  
            trigger = trigger.expand(image.shape[0], -1, -1)


        image = image.clone().detach()

        image[:, start_x:start_x+h, start_y:start_y+w] = trigger
        return image

    def attack_before_train(self, images, labels):
        total_samples = len(labels)
        # poisoned_samples_num = math.floor(total_samples * self.args.poison_ratio)
        poisoned_samples_num = max(1, int(total_samples * self.args.poison_ratio)) 


        if self.from_label is None:
            poisoned_samples_idx = random.sample(range(total_samples), k=poisoned_samples_num)
        else:
            poisoned_samples_idx = np.where(labels.cpu().numpy() == self.from_label)[0]
            if len(poisoned_samples_idx) > poisoned_samples_num:
                poisoned_samples_idx = poisoned_samples_idx[:poisoned_samples_num]

        # 修改数据
        for idx in poisoned_samples_idx:
            labels[idx] = self.to_label
            images[idx] = self.replace_data(images[idx], self.trigger, self.start_x, self.start_y)

        return images, labels

    def train(self, net):
        net.train()
        shared_weights = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.Adam(net.parameters())

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.args.attack_mode == "poison":
                    self.poison_attack = True
                    self.attacker_flag = True
                    images, labels = self.attack_before_train(images, labels)

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        trained_weights = copy.deepcopy(net.state_dict())
        scale_up = 10  
        attack_weights = copy.deepcopy(shared_weights)
        for key in shared_weights.keys():
            difference = trained_weights[key] - shared_weights[key]
            attack_weights[key] += scale_up * difference

        return attack_weights, sum(epoch_loss) / len(epoch_loss), self.attacker_flag

class LocalUpdateScaBack(object):
    def __init__(self, args, dataset, idxs, user_idx, trigger=np.ones([3, 3]), start_x=0, start_y=0, scale_ratio=1.0):
        self.args = args
        self.loss_func = CrossEntropyLoss()
        self.selected_clients = []
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.user_idx = user_idx
        self.poison_attack = False
        self.attacker_flag = False
        self.trigger = trigger
        self.start_x = start_x
        self.start_y = start_y
        self.from_label = None  
        self.to_label = 0  
        self.scale_ratio = scale_ratio  
    
    def replace_data(self, image, trigger, start_x, start_y):

        h, w = trigger.shape

        if isinstance(trigger, np.ndarray):
            trigger = torch.tensor(trigger, dtype=image.dtype, device=image.device)

        if image.dim() == 3: 
            trigger = trigger.expand(image.shape[0], -1, -1)

        image = image.clone().detach()

        image[:, start_x:start_x+h, start_y:start_y+w] = trigger
        return image

    def attack_before_train(self, images, labels):

        total_samples = len(labels)

        poisoned_samples_num = max(1, int(total_samples * self.args.poison_ratio))  #

        if self.from_label is None:
            poisoned_samples_idx = random.sample(range(total_samples), k=poisoned_samples_num)
        else:
            poisoned_samples_idx = np.where(labels.cpu().numpy() == self.from_label)[0]
            if len(poisoned_samples_idx) > poisoned_samples_num:
                poisoned_samples_idx = poisoned_samples_idx[:poisoned_samples_num]

        for idx in poisoned_samples_idx:
            labels[idx] = self.to_label  
            images[idx] = self.replace_data(images[idx], self.trigger, self.start_x, self.start_y) 

        return images, labels

    def train(self, net, global_info=None):

        net.train()
        shared_weights = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.Adam(net.parameters())

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if self.args.attack_mode == "poison":
                    self.poison_attack = True
                    self.attacker_flag = True
                    images, labels = self.attack_before_train(images, labels) 

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                for param in net.parameters():
                    if param.grad is not None:
                        param.grad *= self.scale_ratio
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        trained_weights = copy.deepcopy(net.state_dict())

        return trained_weights, sum(epoch_loss) / len(epoch_loss), self.attacker_flag