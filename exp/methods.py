import copy
import torch
import sys


class NormalDevice():
    def __init__(self, center_device, device_id, trainset, data_idxs, lr=0.01,
                 milestones=None, batch_size=128):
        if milestones == None:
            milestones = [25, 50, 75]

        self.center_device = center_device
        self.net = copy.deepcopy(center_device.net)
        self.id = device_id
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9,
                                         weight_decay=5e-4)
        # optimizer = torch.optim.Adam(device_net.parameters(), lr=lr,
        #                             weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=milestones,
                                                              gamma=0.1)
        device_trainset = DatasetSplit(trainset, data_idxs)
        self.dataloader = torch.utils.data.DataLoader(device_trainset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=2)

        self.base_criterion = torch.nn.CrossEntropyLoss()

        self.train_loss_tracker = []
        self.train_data_loss_tracker = []
        self.train_fisher_loss_tracker = []
        self.train_acc_tracker = []
        self.test_loss_tracker = []
        self.test_data_loss_tracker = []
        self.test_fisher_loss_tracker = []
        self.test_acc_tracker = []


        self.importance = None
        self.reg_coef = None
        
    @property    
    def params_has_gradient(self):
        return {n: p for n, p in self.net.named_parameters()
                                    if p.requires_grad}

    def criterion(self):
        raise NotImplementedError
    


class L2Device(NormalDevice):   
    
    @staticmethod 
    def update_importance(device):
        # Use an identity importance so it is an L2 regularization.
        device.importance = {}
        for n, p in device.params_has_gradient.items():
            device.importance[n] = p.clone().detach().fill_(1.)  # Identity
            device.importance[n].requires_grad = False
            
            
    def update_own_importance(self):
        self.update_importance(self)
            

    def criterion(self, inputs, targets, regularization=False):
        data_loss = self.base_criterion(inputs, targets)

        if regularization:
            reg_loss = 0
            for (n, p), (n_c, p_c) in \
                zip(self.params_has_gradient.items(), \
                    self.center_device.params_has_gradient.items()):
                
                assert n == n_c
                reg_loss += (self.center_device.importance[n] *
                             (p - p_c.detach()) ** 2).sum()
        else:
            reg_loss = torch.tensor([0.0])
        return data_loss + self.reg_coef * reg_loss, (data_loss, reg_loss)
    
    
    def train_one_round(self, epochs):
        
        
        
        # training of this round
        round_train_acc               = []
        round_train_loss_tracker      = []
        round_train_data_loss_tracker = []
        round_train_reg_loss_tracker  = []
        for epoch in range(epochs):
            acc, \
            train_loss_tracker, \
            train_data_loss_tracker, \
            train_reg_loss_tracker = \
            self.train(epoch)
            
            round_acc.append(acc)
            round_train_loss_tracker.extend(train_loss_tracker)
            round_train_data_loss_tracker.extend(train_data_loss_tracker)
            round_train_reg_loss_tracker.extend(train_reg_loss_tracker)
             
        self.train_acc_tracker.append(copy.deepcopy(round_train_acc))
        self.train_loss_tracker.append(copy.deepcopy(round_train_loss_tracker))
        self.train_data_loss_tracker.append(copy.deepcopy(round_train_data_loss_tracker))
        self.train_reg_loss_tracker.append(copy.deepcopy(round_train_reg_loss_tracker))
        
        self.update_importance_before_round_flag == False
        self.update_previous_before_round_flag == False
            
        
        
        

    def train(self, epoch):
        train_loss_tracker = []
        train_data_loss_tracker = []
        train_reg_loss_tracker = []
        
        net.train()
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss, data_loss, reg_loss = self.criterion(
                outputs, targets, regularization=True)

            loss.backward()
            # nn.utils.clip_grad_norm_(device['net'].parameters(), max_norm = 1)
            # nn.utils.clip_grad_value_(device['net'].parameters(), clip_value=1)

            self.optimizer.step()
            train_loss += loss.item()
            train_loss_tracker.append(loss.item())
            train_data_loss_tracker.append(data_loss.item())
            train_reg_loss_tracker.append(reg_loss.item())
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        avg_loss = train_loss / (batch_idx + 1)
        acc = 100. * correct / total
        
        print(f'\r(Device {self.id}/Epoch {epoch}) ' +
                         f'Train Loss: {avg_loss:.3f}, ' +
                         f'Data: {data_loss.item():.3f} Reg: {reg_loss.item():.3f}' +
                         f'| Train Acc: {acc:.3f}')
        
        return acc, train_loss_tracker, train_data_loss_tracker, train_reg_loss_tracker


    def test(self, epoch):
        net.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = device['net'](inputs)
                
                if isinstance(device['criterion'], nn.CrossEntropyLoss):
                    loss = device['criterion'](outputs, targets)
                else:
                    loss, data_loss, fisher_loss = device['criterion'](outputs, targets)
                
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                avg_loss = test_loss / (batch_idx + 1)
                acc = 100.* correct / total
                
        device['test_loss_tracker'].append(avg_loss)
        sys.stdout.write(f' | Test Loss: {avg_loss:.3f} | Test Acc: {acc:.3f}\n')
        sys.stdout.flush()  
        acc = 100.*correct/total
        device['test_acc_tracker'].append(acc)
        return acc




class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, torch.tensor(label)
