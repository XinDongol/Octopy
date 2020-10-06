import glob, os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from utils import progress_bar

from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--baseline', type=int, default=0)
parser.add_argument('--T', type=float, default=20.)
parser.add_argument('--lamb', type=float, default=0.5)
parser.add_argument('--kd_scale', type=float, default=1.)
args = parser.parse_args()
print(vars(args))

writer = SummaryWriter(args.root)

with open(args.root+'/commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    
    

file_name = glob.glob('/project/kung/xin/cifar_vgg11_saved_model/gamma_beta_inverted_images/*.pth')

real_inp_l = []
real_target_l = []
fake_inp_l = []
for i in file_name:
    a = torch.load(i)
    real_inp_l.append(a['real_inp'])
    real_target_l.append(a['real_label'])
    fake_inp_l.append(a['fake_inp'])
    del a

real_inp = torch.cat(real_inp_l)
real_target = torch.cat(real_target_l)
fake_inp = torch.cat(fake_inp_l)

kd_dataset = torch.utils.data.TensorDataset(real_inp, real_target, fake_inp)

kd_loader = torch.utils.data.DataLoader(
    kd_dataset, batch_size=128, shuffle=True, num_workers=8)

# Preparing data..
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/dev/shm', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=64, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='/dev/shm', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def test(net, testloader, on_fake=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        if isinstance(testloader.dataset, torch.utils.data.TensorDataset):
            for batch_idx, (real_inp, real_target, fake_inp) in enumerate(testloader):
                if on_fake:
                    inputs, targets = fake_inp, real_target
                else:
                    inputs, targets = real_inp, real_target
                
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        else:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

    test_loss = test_loss/(batch_idx+1)
    acc = 100.*correct/total
    return test_loss, acc


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    
device = 'cuda'
criterion = nn.CrossEntropyLoss()

teacher = VGG('VGG11')
teacher.load_state_dict(torch.load('./cifar_vgg_11.pth')['net'])
teacher = teacher.to(device)
teacher.eval()

teacher_test_loss, teacher_test_acc = test(teacher, testloader, on_fake=None)
print('Teacher Test on Test Loss: %.4f | Acc %.2f'%(teacher_test_loss, teacher_test_acc))

teacher_test_loss, teacher_test_acc = test(teacher, kd_loader, on_fake=True)
print('Teacher Test on Fake Loss: %.4f | Acc %.2f'%(teacher_test_loss, teacher_test_acc))

teacher_test_loss, teacher_test_acc = test(teacher, kd_loader, on_fake=False)
print('Teacher Test on Train Loss: %.4f | Acc %.2f'%(teacher_test_loss, teacher_test_acc))


student = VGG('VGG11')
student = student.to(device)
student.eval()
student_start_test_loss, stduent_start_test_acc = test(student, testloader)
print('Student Start Test on Test Loss: %.4f | Acc %.2f'%(student_start_test_loss, stduent_start_test_acc))


optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)


def kd_loss(out_s, out_t, T, lamb, kd_scale, target=None):
    batch_size = out_s.size(0)
    s_max = F.log_softmax(out_s/T, dim=1)
    t_max = F.softmax(out_t/T, dim=1)
    loss_kd = F.kl_div(input=s_max, target=t_max, size_average=False) / batch_size
    if target is not None:
        standard_loss = F.cross_entropy(input=out_s, target=target)
    else:
        standard_loss = 0.
        lamb = 1.0
    loss = (1-lamb)*standard_loss + lamb*T*T*loss_kd*kd_scale
    
    return loss
    

# Training
def kd_train(kd_loader, teacher, student, T, lamb, kd_scale, baseline):
    teacher.eval()
    student.train()
    
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (real_inp, real_target, fake_inp) in enumerate(kd_loader):
        real_inp, real_target, fake_inp = real_inp.to(device), real_target.to(device), fake_inp.to(device)
        optimizer.zero_grad()
        
        if baseline:
            out_t = teacher(real_inp)
            out_s = student(real_inp)
            loss = kd_loss(out_s, out_t, T, lamb, kd_scale, real_target)
        else:
            out_t = teacher(fake_inp)
            out_s = student(fake_inp)
            loss = kd_loss(out_s, out_t, T, lamb, kd_scale, real_target)            
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = out_s.max(1)
        total += real_target.size(0)
        correct += predicted.eq(real_target).sum().item()

        progress_bar(batch_idx, len(kd_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    return train_loss/(batch_idx+1), 100.*correct/total





for epoch_idx in range(args.epochs):
    train_loss, train_acc = kd_train(kd_loader, teacher, student, args.T, args.lamb, args.kd_scale, args.baseline)
    test_loss, test_acc   = test(student, testloader, None)
    writer.add_scalar('train/train_loss', train_loss, epoch_idx)
    writer.add_scalar('train/train_acc', train_acc, epoch_idx)
    writer.add_scalar('test/test_loss', test_loss, epoch_idx)
    writer.add_scalar('test/test_acc', test_acc, epoch_idx)