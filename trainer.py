########################################
#### Licensed under the MIT license ####
########################################

import torch
import torch.nn as nn
import torch.optim as optim
import os

import numpy as np
import cv2

from numpy import prod
from datetime import datetime
from model import CapsuleNetwork
from loss import CapsuleLoss
from time import time
from torchsummary import summary

SAVE_MODEL_PATH = 'checkpoints/'
if not os.path.exists(SAVE_MODEL_PATH):
    os.mkdir(SAVE_MODEL_PATH)

class CapsNetTrainer:
    """
    Wrapper object for handling training and evaluation
    """
    def __init__(self, loaders, batch_size, learning_rate, num_routing=3, lr_decay=0.99, classes=7, num_filters=128, stride=2, filter_size=5, recons=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), multi_gpu=(torch.cuda.device_count() > 1)):
        self.device = device
        self.multi_gpu = multi_gpu
        self.recons = recons
        self.classes = classes
        self.loaders = loaders
        img_shape = self.loaders['train'].dataset[0][0].numpy().shape
        
        self.net = CapsuleNetwork(img_shape, num_filters, stride, filter_size, recons, primary_dim=8, num_classes=self.classes, out_dim=16, num_routing=num_routing, device=self.device).to(self.device)
        #summary(self.net, (3, 70, 70))
        if self.multi_gpu:
            self.net = nn.DataParallel(self.net)

        self.criterion = CapsuleLoss(recons, loss_lambda=0.5, recon_loss_scale=5e-4)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        print(8*'#', 'PyTorch Model built'.upper(), 8*'#')
        print('Num params:', sum([prod(p.size()) for p in self.net.parameters()]))
    
    def __repr__(self):
        return repr(self.net)

    def run(self, epochs, classes):
        print(8*'#', 'Run started'.upper(), 8*'#')
        eye = torch.eye(len(classes)).to(self.device)
        
        for epoch in range(1, epochs+1):
            for phase in ['train', 'eval']:
                print(f'{phase}ing...'.capitalize())
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                t0 = time()
                running_loss = 0.0
                correct = 0; total = 0
                for i, (images, labels) in enumerate(self.loaders[phase]):
                    t1 = time()
                    images, labels = images.to(self.device), labels.to(self.device)
                    # One-hot encode labels
                    labels = eye[labels]

                    self.optimizer.zero_grad()

                    outputs = self.net(images)
                    if type(outputs) is tuple:
                        loss = self.criterion(outputs[0], labels, images, outputs[1])
                    else:
                        loss = self.criterion(outputs, labels, images, None)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    _, labels = torch.max(labels, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    accuracy = float(correct) / float(total)

                    if phase == 'train':
                        print(f'Epoch {epoch}, Batch {i+1}, Loss {running_loss/(i+1)}',
                        f'Accuracy {accuracy} Time {round(time()-t1, 3)}s')
                
                print(f'{phase.upper()} Epoch {epoch}, Loss {running_loss/(i+1)}',
                f'Accuracy {accuracy} Time {round(time()-t0, 3)}s')
            
            self.scheduler.step()
            
        now = str(datetime.now()).replace(" ", "-")
        error_rate = round((1-accuracy)*100, 2)
        torch.save(self.net.state_dict(), os.path.join(SAVE_MODEL_PATH, f'{error_rate}_{now}.pth.tar'))

        class_correct = list(0. for _ in classes)
        class_total = list(0. for _ in classes)
        correct = 0
        total = 0
        matrix = np.zeros((7, 7), dtype=np.int)
        for images, labels in self.loaders['test']:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.net(images)
            if type(outputs) is tuple:
                outputs = outputs[0]
            # image = np.array(((reconstructions[0].cpu().detach().numpy() * 0.5) + 0.5) * 255, dtype=np.int32)
            # image = np.moveaxis(image, 0, -1)
            # image_gt = np.array(((images[0].cpu().detach().numpy() * 0.5) + 0.5) * 255, dtype=np.int32)
            # image_gt = np.moveaxis(image_gt, 0, -1)
            # cv2.imwrite("/home/bax/Data/Dumpsters/capsule-network/" + str(epoch) + ".jpg", image)
            # cv2.imwrite("/home/bax/Data/Dumpsters/capsule-network/" + str(epoch) + "_gt.jpg", image_gt)
            _, predicted = torch.max(outputs, 1)
            labels = eye[labels]
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            accuracy = float(correct) / float(total)
            for i in range(labels.size(0)):
                matrix[labels[i], predicted[i]] += 1
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        print("Test accuracy", accuracy)
        print(matrix)

        for i in range(len(classes)):
            print('Accuracy of %5s : %4f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
