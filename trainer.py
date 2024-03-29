########################################
#### Licensed under the MIT license ####
########################################

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from numpy import prod
from model import CapsuleNetwork
from loss import CapsuleLoss
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary

SAVE_MODEL_PATH = 'checkpoints/'
if not os.path.exists(SAVE_MODEL_PATH):
    os.mkdir(SAVE_MODEL_PATH)


class CapsNetTrainer:
    """
    Wrapper object for handling training and evaluation
    """

    def __init__(self, loaders, learning_rate, num_routing=3, lr_decay=0.99, labels=8, num_filters=128, stride=2,
                 filter_size=5, recons=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 multi_gpu=(torch.cuda.device_count() > 1)):
        self.device = device
        self.multi_gpu = multi_gpu
        self.recons = recons
        self.labels = labels
        self.loaders = loaders
        img_shape = self.loaders['train'].dataset[0][0].numpy().shape

        self.net = CapsuleNetwork(img_shape, num_filters, stride, filter_size, recons, primary_dim=8,
                                  num_labels=self.labels, out_dim=16, num_routing=num_routing, device=self.device).to(
            self.device)
        summary(self.net, (3, 70, 70))
        if self.multi_gpu:
            self.net = nn.DataParallel(self.net)

        self.criterion = CapsuleLoss(recons, loss_lambda=0.5, recon_loss_scale=5e-4)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        print(8 * '#', 'PyTorch Model built'.upper(), 8 * '#')
        print('Num params:', sum([prod(p.size()) for p in self.net.parameters()]))

    def __repr__(self):
        return repr(self.net)

    def run(self, epochs, labels_name, num_labels):
        print(8 * '#', 'Run started'.upper(), 8 * '#')
        for epoch in range(1, epochs + 1):
            for phase in ['train', 'eval']:
                print(f'{phase}ing...'.capitalize())
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                t0 = time()
                running_loss = 0.0
                correct = 0
                total = 0
                accuracy = 0
                for i, (images, labels) in enumerate(self.loaders[phase]):
                    t1 = time()
                    images, labels = images.to(self.device), labels.to(self.device)
                    # One-hot encode labels
                    # labels = [eye[labels[0]], labels[1:]]

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

                    predicted = torch.round(outputs[0]) if type(outputs) is tuple else torch.round(outputs)
                    predicted = predicted.long()
                    total += (labels.size(0) * labels.size(1))
                    correct += (predicted == labels).sum()
                    accuracy = float(correct) / float(total)

                    if phase == 'train':
                        print(f'Epoch {epoch}, Batch {i + 1}, Loss {running_loss / (i + 1)}',
                              f'Accuracy {accuracy}, Time {round(time() - t1, 3)}s')

                print(f'{phase.upper()} Epoch {epoch}, Loss {running_loss / (i + 1)}',
                      f'Accuracy {accuracy}, Time {round(time() - t0, 3)}s')

            self.scheduler.step()

        # now = str(datetime.now()).replace(" ", "-")
        # error_rate = round((1 - accuracy) * 100, 2)
        torch.save(self.net.state_dict(), os.path.join(SAVE_MODEL_PATH, 'modelo.pth.tar'))

        class_correct = list(0. for _ in labels_name)
        class_total = list(0. for _ in labels_name)
        correct = 0
        total = 0
        matrices = []  # confusion matrices
        for i in range(len(num_labels)):
            matrices.append(np.zeros((num_labels[i], num_labels[i]), dtype=np.int))

        first_batch = True
        for images, labels in self.loaders['test']:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.net(images)
            if type(outputs) is tuple:
                if first_batch:
                    for index, image in enumerate(outputs[1]):
                        plt.imshow(image.permute(1, 2, 0).cpu().detach(), vmin=0, vmax=1)
                        plt.savefig(f"{index}.png")
                        plt.imshow(images[index].permute(1, 2, 0).cpu().detach(), vmin=0, vmax=1)
                        plt.savefig(f"original{index}.png")
                first_batch = False
                outputs = outputs[0]

            predicted = torch.round(outputs)
            predicted = predicted.long()
            # labels = eye[labels]
            # _, labels = torch.max(labels, 1)
            total += (labels.size(0) * labels.size(1))
            correct += (predicted == labels).sum()
            for i in range(labels.size(0)):
                for j in range(labels.size(1)):
                    if labels[i, j] != -1:
                        matrices[j][labels[i, j], predicted[i, j]] += 1

            for i in range(labels.size(0)):
                c = (predicted[i] == labels[i]).squeeze()
                for j in range(labels.size(1)):
                    label = j
                    class_correct[label] += c[j].item()
                    class_total[label] += 1

        accuracy = float(correct) / float(total)
        print(f"Test accuracy {accuracy}")

        for i in range(len(labels_name)):
            print(matrices[i])

            plt.figure(figsize=(10, 10), dpi=100)
            plt.tight_layout()
            sns.set(font_scale=1.1)
            ax = sns.heatmap(matrices[i], annot=True, fmt='d')
            ax.xaxis.set_ticklabels(['False', 'True'])
            ax.set_ylabel("Real", fontsize=14, labelpad=20)
            ax.yaxis.set_ticklabels(['False', 'True'])
            ax.set_title(f"Confusion Matrix for {labels_name[i]}", fontsize=14, pad=20)

            if class_total[i] != 0:
                if num_labels[i] == 2:  # Resultado binario, calcular cuando es True
                    precision = matrices[i][1, 1] / (matrices[i][1, 1] + matrices[i][0, 1])
                    recall = matrices[i][1, 1] / (matrices[i][1, 1] + matrices[i][1, 0])

                else:
                    precisions = []
                    recalls = []

                    for actual_class in range(num_labels[i]):
                        precisions.append(matrices[i][actual_class, actual_class] / (
                                matrices[i][actual_class, actual_class] +
                                matrices[i][actual_class, :actual_class].sum() +
                                matrices[i][actual_class, actual_class + 1:].sum()))

                        recalls.append(matrices[i][actual_class, actual_class] / (
                                matrices[i][actual_class, actual_class] +
                                matrices[i][:actual_class, actual_class].sum() +
                                matrices[i][actual_class + 1:, actual_class].sum()))

                    precision = np.nanmean(precisions)
                    recall = np.nanmean(recalls)

                print('Accuracy of %5s : %4f %%' % (
                    labels_name[i], 100 * class_correct[i] / class_total[i]))
                print('Precision of %5s : %4f %%' % (labels_name[i], 100 * precision))
                print('Recall of %5s : %4f %%' % (labels_name[i], 100 * recall))
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}".format(
                    accuracy, precision, recall)
                ax.set_xlabel("Predicted" + stats_text, fontsize=14, labelpad=20)

            else:
                ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
            plt.savefig(os.path.join(SAVE_MODEL_PATH, f'{labels_name[i]}.png'))
