import os
import torch
from torchvision import transforms, datasets
from trainer import CapsNetTrainer
import argparse

DATA_PATH = 'data'

# Collect arguments (if any)
parser = argparse.ArgumentParser()

# MNIST or CIFAR?
parser.add_argument('dataset', nargs='?', type=str, default='DUMPSTERS', help="'DUMPSTERS'")
# Batch size
parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size.')
# Epochs
parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs.')
# Number of filters
parser.add_argument('-nf', '--num_filters', type=int, default=128, help='Number of filters per layer.')
# Stride
parser.add_argument('-s', '--stride', type=int, default=2, help='Stride of the convolutional layers.')
# Filter size
parser.add_argument('-fs', '--filter_size', type=int, default=5, help='Size of the convolutional filters.')
# Reconstruction
parser.add_argument('-recons', '--reconstruction', action='store_true', help='Flag to reconstruction.')
# Crop
parser.add_argument('-c', '--crop', action='store_false', help='Flag to crop.')
# Jitter
parser.add_argument('-j', '--jitter', action='store_true', help='Flag to jitter.')
# Rotation
parser.add_argument('-r', '--rotation', action='store_true', help='Flag to rotation.')
# Flip
parser.add_argument('-f', '--flip', action='store_false', help='Flag to horizontal flip.')
# Learning rate
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate.')
# Number of routing iterations
parser.add_argument('--num_routing', type=int, default=3, help='Number of routing iteration in routing capsules.')
# Exponential learning rate decay
parser.add_argument('--lr_decay', type=float, default=0.96, help='Exponential learning rate decay.')
# Select device "cuda" for GPU or "cpu"
parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=['cuda', 'cpu'], help='Device to use. Choose "cuda" for GPU or "cpu".')
# Use multiple GPUs?
parser.add_argument('--multi_gpu', action='store_true', help='Flag whether to use multiple GPUs.')
# Select GPU device
parser.add_argument('--gpu_device', type=int, default=None, help='ID of a GPU to use when multiple GPUs are available.')
# Data directory
parser.add_argument('--data_path', type=str, default="train7", help='Path to the DUMPSTERS dataset. Alternatively you can set the path as an environmental variable $data.')
args = parser.parse_args()
print(args)
device = torch.device(args.device)

if args.gpu_device is not None:
    torch.cuda.set_device(args.gpu_device)

if args.multi_gpu:
    args.batch_size *= torch.cuda.device_count()

if args.dataset.upper() == 'DUMPSTERS':
    args.train_data_path = "/home/bax/Data/Dumpsters/dataset/" + args.data_path
    args.eval_data_path = "/home/bax/Data/Dumpsters/dataset/eval7/"
    args.test_data_path = "/home/bax/Data/Dumpsters/dataset/test7/"
    size = 299
    num_classes = 7
    classes = list(range(num_classes))
    mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
else:
    raise ValueError('Dataset must be DUMPSTERS')

resize = transforms.Compose([
    transforms.Resize((70, 70))
])

transform = transforms.Compose([
    resize
])

if args.crop:
    transform = transforms.Compose([
        transform,
        transforms.RandomCrop((70, 70), padding=7)
    ])

if args.jitter:
    transform = transforms.Compose([
        transform,
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    ])

if args.flip:
    transform = transforms.Compose([
        transform,
        transforms.RandomHorizontalFlip()
    ])

if args.rotation:
    transform = transforms.Compose([
        transform,
        transforms.RandomRotation(10)
    ])

common = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform = transforms.Compose([
    transform,
    common
])

transform_eval_test = transforms.Compose([
    resize,
    common
])


loaders = {}
if args.dataset.upper() == 'DUMPSTERS':
    trainset = datasets.ImageFolder(root=args.train_data_path, transform=transform)
    evalset = datasets.ImageFolder(root=args.eval_data_path, transform=transform_eval_test)
    testset = datasets.ImageFolder(root=args.test_data_path, transform=transform_eval_test)
loaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
loaders['eval'] = torch.utils.data.DataLoader(evalset, batch_size=args.batch_size, shuffle=False, num_workers=2)
loaders['test'] = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
print(8*'#', f'Using {args.dataset.upper()} dataset', 8*'#')

# Run
caps_net = CapsNetTrainer(loaders, args.batch_size, args.learning_rate, args.num_routing, args.lr_decay, num_classes, args.num_filters, args.stride, args.filter_size, args.reconstruction, device=device, multi_gpu=args.multi_gpu)
caps_net.run(args.epochs, classes=classes)
