import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
from models.models_for_cub import ResNet
from cub import cub200
import os
import matplotlib.pyplot as plt
import shutil
from utils.Config import Config
class NetworkManager(object):
    def __init__(self, options, path):
        self.options = options
        self.path = path
        self.device = options['device']

        print('Starting to prepare network and data...')

        self.net = nn.DataParallel(self._net_choice(self.options['net_choice'])).to(self.device)
        #self.net.load_state_dict(torch.load('/home/zhangyongshun/se_base_model/model_save/ResNet/backup/epoch120/ResNet50-finetune_fc_cub.pkl'))
        print('Network is as follows:')
        print(self.net)
        #print(self.net.state_dict())
        self.criterion = nn.CrossEntropyLoss()
        self.solver = torch.optim.SGD(
            self.net.parameters(), lr=self.options['base_lr'], momentum=self.options['momentum'], weight_decay=self.options['weight_decay']
        )
        self.schedule = torch.optim.lr_scheduler.StepLR(self.solver, step_size=30, gamma=0.1)
        #self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.solver, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4
        #)

        train_transform_list = [
            transforms.RandomResizedCrop(self.options['img_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
        test_transforms_list = [
            transforms.Resize(int(self.options['img_size']/0.875)),
            transforms.CenterCrop(self.options['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
        train_data = cub200(self.path['data'], train=True, transform=transforms.Compose(train_transform_list))
        test_data = cub200(self.path['data'], train=False, transform=transforms.Compose(test_transforms_list))
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.options['batch_size'], shuffle=True, num_workers=4, pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )

    def train(self):
        epochs  = np.arange(1, self.options['epochs']+1)
        test_acc = list()
        train_acc = list()
        print('Training process starts:...')
        if torch.cuda.device_count() > 1:
            print('More than one GPU are used...')
        print('Epoch\tTrainLoss\tTrainAcc\tTestAcc')
        print('-'*50)
        best_acc = 0.0
        best_epoch = 0
        self.net.train(True)
        for epoch in range(self.options['epochs']):
            num_correct = 0
            train_loss_epoch = list()
            num_total = 0
            for imgs, labels in self.train_loader:
                self.solver.zero_grad()
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.net(imgs)
                loss = self.criterion(output, labels)
                _, pred = torch.max(output, 1)
                num_correct += torch.sum(pred == labels.detach_())
                num_total += labels.size(0)
                train_loss_epoch.append(loss.item())
                loss.backward()
                #nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.solver.step()

            train_acc_epoch = num_correct.detach().cpu().numpy()*100 / num_total
            avg_train_loss_epoch  = sum(train_loss_epoch)/len(train_loss_epoch)
            test_acc_epoch = self._accuracy()
            test_acc.append(test_acc_epoch)
            train_acc.append(train_acc_epoch)
            self.schedule.step()
            if test_acc_epoch>best_acc:
                best_acc = test_acc_epoch
                best_epoch = epoch+1
                print('*', end='')
                #torch.save(self.net.state_dict(), os.path.join(self.path['model_save'], self.options['net_choice'], self.options['net_choice']+str(self.options['model_choice'])+'.pkl'))
            print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%'.format(epoch+1, avg_train_loss_epoch, train_acc_epoch, test_acc_epoch))
        plt.figure()
        plt.plot(epochs, test_acc, color='r', label='Test Acc')
        plt.plot(epochs, train_acc, color='b', label='Train Acc')

        plt.xlabel('epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.title(self.options['net_choice']+str(self.options['model_choice']))
        plt.savefig(self.options['net_choice']+str(self.options['model_choice'])+'.png')

    def _accuracy(self):
        self.net.eval()
        num_total = 0
        num_acc = 0
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.net(imgs)
                _, pred = torch.max(output, 1)
                num_acc += torch.sum(pred==labels.detach_())
                num_total += labels.size(0)
        return num_acc.detach().cpu().numpy()*100/num_total

    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def _net_choice(self, net_choice):
        if net_choice=='ResNet':
            return ResNet(pre_trained=True, n_class=200, model_choice=self.options['model_choice'])
        elif net_choice=='ResNet_ED':
            return ResNet_ED(pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=self.options['model_choice'])
        elif net_choice == 'ResNet_SE':
            return ResNet_SE(pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=self.options['model_choice'])
        elif net_choice == 'ResNet_self':
            return ResNet_self(pre_trained=True, pre_trained_weight_gpu=True, n_class=200, model_choice=self.options['model_choice'])

    def adjust_learning_rate(optimizer, epoch, args):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
