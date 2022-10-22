import torch
from torchvision import transforms
from trainer import CapsNetTrainer
import argparse
import dataset

# Collect arguments (if any)
parser = argparse.ArgumentParser()
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
parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                    choices=['cuda', 'cpu'], help='Device to use. Choose "cuda" for GPU or "cpu".')
# Use multiple GPUs?
parser.add_argument('--multi_gpu', action='store_true', help='Flag whether to use multiple GPUs.')
# Select GPU device
parser.add_argument('--gpu_device', type=int, default=None, help='ID of a GPU to use when multiple GPUs are available.')
# Data directory
parser.add_argument('--data_path', type=str, default="train7",
                    help='Path to the DUMPSTERS dataset. Alternatively you can set the path as an environmental '
                         'variable $data.')
args = parser.parse_args()
print(args)
device = torch.device(args.device)

if args.gpu_device is not None:
    torch.cuda.set_device(args.gpu_device)

if args.multi_gpu:
    args.batch_size *= torch.cuda.device_count()


args.train_data_path = "C:/Users/elena/Documents/TFM/AA_TAG_IMAGENES/train.csv"
args.eval_data_path = "C:/Users/elena/Documents/TFM/AA_TAG_IMAGENES/validation.csv"
args.test_data_path = "C:/Users/elena/Documents/TFM/AA_TAG_IMAGENES/test.csv"
size = 299
mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
labels = ["TIPO_CONTENEDOR", "SOTERRADO", "ESTADO_GRAFITI", "ESTADO_QUEMADO", "ESTADO_N_BOCA", "ESTADO_BOCAS",
          "ESTADO_CIERRE", "ESTADO_TIPO_DE_CIERRE", "ESTADO_SERIGRAFIA", "ESTADO_CUERPO",
          "ESTADO_TAPA", "ESTADO_ELEMENTOS_DE_EVALUACION", "ESTADO_TIPO_DE_CARGA"]
dictionary = {"TIPO_CONTENEDOR": {"0": "Envases Ligeros",
                                  "1": "Materia Orgánica",
                                  "2": "Papel Cartón",
                                  "3": "Resto",
                                  "4": "Vidrio"},
              "SOTERRADO": {"0": "N", "1": "S"},
              "ESTADO_GRAFITI": {"0": "N", "1": "S"},
              "ESTADO_QUEMADO": {"0": "N", "1": "S"},
              "ESTADO_BOCAS": {"0": "Sin bocas", "1": "Defectuosas", "2": "Buenas"},
              "ESTADO_CIERRE": {"0": "Sin cierre", "1": "Roto/Estropeado", "2": "Integro"},
              "ESTADO_TIPO_DE_CIERRE": {"0": "Sin cierre", "1": "Barra bloqueo", "2": "Candado", "3": "Cierre",
                                        "4": "Frontal metálico por gravedad", "5": "Frontal plástico", "6": "Iglú",
                                        "7": "Lateral metálico por gravedad", "8": "Otros"},
              "ESTADO_SERIGRAFIA": {"0": "Sin serigrafía", "1": "Incompleta", "2": "Completa"},
              "ESTADO_CUERPO": {"1": "Defectuoso", "0": "Correcto"},
              "ESTADO_TAPA": {"1": "Defectuoso", "0": "Correcto"},
              "ESTADO_ELEMENTOS_DE_EVALUACION": {"1": "Defectuoso", "0": "Correcto"},
              "ESTADO_TIPO_DE_CARGA": {"0": "Trasera", "1": "Lateral", "2": "Superior"}}

num_labels = len(labels)
num_classes = [5, 2, 2, 2, 5, 3, 3, 9, 3, 2, 2, 2, 3]

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
# trainset = datasets.ImageFolder(root=args.train_data_path, transform=transform)
# evalset = datasets.ImageFolder(root=args.eval_data_path, transform=transform_eval_test)
# testset = datasets.ImageFolder(root=args.test_data_path, transform=transform_eval_test)
columns = ["FOTO_CONTENEDOR", "TIPO_CONTENEDOR", "SOTERRADO", "ESTADO_GRAFITI", "ESTADO_QUEMADO", "ESTADO_N_BOCA",
           "ESTADO_BOCAS", "ESTADO_CIERRE", "ESTADO_TIPO_DE_CIERRE", "ESTADO_SERIGRAFIA", "ESTADO_CUERPO",
           "ESTADO_TAPA", "ESTADO_ELEMENTOS_DE_EVALUACION", "ESTADO_TIPO_DE_CARGA"]
trainset = dataset.DumpstersDataset(args.train_data_path, columns=columns, transform=transform)
evalset = dataset.DumpstersDataset(args.eval_data_path, columns=columns, transform=transform_eval_test)
testset = dataset.DumpstersDataset(args.test_data_path, columns=columns, transform=transform_eval_test)
# trainset = datasets.MNIST('/files/', train=True, download=True, transform=transform)
# evalset = datasets.MNIST('/files/', train=False, download=True, transform=transform_eval_test)
# testset = datasets.MNIST('/files/', train=False, download=True, transform=transform_eval_test)

loaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
loaders['eval'] = torch.utils.data.DataLoader(evalset, batch_size=args.batch_size, shuffle=False)
loaders['test'] = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

print(8 * '#', 'Using Dumpster dataset', 8 * '#')

# Run
caps_net = CapsNetTrainer(loaders, args.batch_size, args.learning_rate, args.num_routing, args.lr_decay, num_labels,
                          args.num_filters, args.stride, args.filter_size, args.reconstruction, device=device,
                          multi_gpu=args.multi_gpu)
caps_net.run(args.epochs, labels_name=labels, num_classes=num_classes)
