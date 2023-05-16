import torch
from torchvision import transforms
from trainer import CapsNetTrainer
import argparse
import dataset


class Main:
    def __init__(self, arguments):
        self.dataset = arguments.dataset
        self.batch_size = arguments.batch_size
        self.learning_rate = arguments.learning_rate
        self.lr_decay = arguments.lr_decay
        self.num_routing = arguments.num_routing
        self.num_filters = arguments.num_filters
        self.device = torch.device(arguments.device)
        self.stride = arguments.stride
        self.filter_size = arguments.filter_size
        self.reconstruction = arguments.reconstruction
        self.multi_gpu = arguments.multi_gpu
        self.epochs = arguments.epochs

        if arguments.gpu_device is not None:
            torch.cuda.set_device(arguments.gpu_device)

        if self.multi_gpu:
            self.batch_size *= torch.cuda.device_count()

        train_data_path = f"{arguments.data_folder}/train.csv"
        eval_data_path = f"{arguments.data_folder}/validation.csv"
        test_data_path = f"{arguments.data_folder}/test.csv"

        if self.dataset.upper() == 'CONTENEDOR':
            self.labels = ["ENVASES", "ORGANICA", "PAPEL_CARTON", "RESTOS", "VIDRIO", "SOTERRADO", "ESTADO_GRAFITI",
                           "ESTADO_QUEMADO"]
            self.num_labels = len(self.labels)
            self.num_classes = [2, 2, 2, 2, 2, 2, 2, 2]
            columns = ["FOTO_CONTENEDOR", "ENVASES", "ORGANICA", "PAPEL_CARTON", "RESTOS", "VIDRIO", "SOTERRADO",
                       "ESTADO_GRAFITI",
                       "ESTADO_QUEMADO"]
        elif self.dataset.upper() == 'PUNTO':
            self.labels = ['ENT_BOLSAS_EELL', 'ENT_PAPEL_CARTON_DOMESTICO', 'ENT_PAPEL_CARTON_INDUSTRIAL',
                           'ENT_VIDRIO', 'ENT_BOLSAS_RESTO', 'ENT_PODAS', 'ENT_ESCOMBROS', 'ENT_OBJETOS_VOLUMINOSOS',
                           'ENT_OTROS', 'DESBORDE_RESTO', 'DESBORDE_EELL', 'DESBORDE_PC', 'DESBORDE_VIDRIO',
                           'ESTABL_COMERCIOS',
                           'ESTABL_HORECA', 'ESTABL_OBRAS', 'ESTABL_CENTROPUBLICO']
            self.num_labels = len(self.labels)
            self.num_classes = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            columns = ['FOTO_PUNTO', 'ENT_BOLSAS_EELL', 'ENT_PAPEL_CARTON_DOMESTICO', 'ENT_PAPEL_CARTON_INDUSTRIAL',
                       'ENT_VIDRIO', 'ENT_BOLSAS_RESTO', 'ENT_PODAS', 'ENT_ESCOMBROS', 'ENT_OBJETOS_VOLUMINOSOS',
                       'ENT_OTROS', 'DESBORDE_RESTO', 'DESBORDE_EELL', 'DESBORDE_PC', 'DESBORDE_VIDRIO',
                       'ESTABL_COMERCIOS',
                       'ESTABL_HORECA', 'ESTABL_OBRAS', 'ESTABL_CENTROPUBLICO']
        else:
            raise ValueError('Dataset must be Punto or Contenedor')

        resize = transforms.Compose([
            transforms.Resize((70, 70))
        ])

        transform = transforms.Compose([
            resize
        ])

        if arguments.crop:
            transform = transforms.Compose([
                transform,
                transforms.RandomCrop((70, 70), padding=7)
            ])

        if arguments.jitter:
            transform = transforms.Compose([
                transform,
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
            ])

        if arguments.flip:
            transform = transforms.Compose([
                transform,
                transforms.RandomHorizontalFlip()
            ])

        if arguments.rotation:
            transform = transforms.Compose([
                transform,
                transforms.RandomRotation(10)
            ])

        mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

        self.loaders = {}
        trainset = dataset.DumpstersDataset(train_data_path, columns=columns, transform=transform)
        evalset = dataset.DumpstersDataset(eval_data_path, columns=columns, transform=transform_eval_test)
        testset = dataset.DumpstersDataset(test_data_path, columns=columns, transform=transform_eval_test)

        self.loaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.loaders['eval'] = torch.utils.data.DataLoader(evalset, batch_size=self.batch_size, shuffle=False)
        self.loaders['test'] = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)

    def main(self):
        print(8 * '#', f'Using {self.dataset} dataset', 8 * '#')

        # Run
        caps_net = CapsNetTrainer(self.loaders, self.learning_rate, self.num_routing, self.lr_decay, self.num_labels,
                                  self.num_filters, self.stride, self.filter_size, self.reconstruction,
                                  device=self.device, multi_gpu=self.multi_gpu)
        caps_net.run(self.epochs, labels_name=self.labels, num_classes=self.num_classes)


if __name__ == "__main__":
    # Collect arguments (if any)
    parser = argparse.ArgumentParser()
    # Punto o Contenedor?
    parser.add_argument('--dataset', type=str, default='Punto', help="type of images: punto or contenedor.")
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
    parser.add_argument('-c', '--crop', action='store_true', help='Flag to crop.')
    # Jitter
    parser.add_argument('-j', '--jitter', action='store_true', help='Flag to jitter.')
    # Rotation
    parser.add_argument('-r', '--rotation', action='store_true', help='Flag to rotation.')
    # Flip
    parser.add_argument('-f', '--flip', action='store_true', help='Flag to horizontal flip.')
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
    parser.add_argument('--gpu_device', type=int, default=None,
                        help='ID of a GPU to use when multiple GPUs are available.')
    # Data directory
    parser.add_argument('--data_folder', type=str, help='Path to the folder containing the datasets')
    args = parser.parse_args()
    print(args)
    main = Main(args)
    main.main()
