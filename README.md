# MultiLabel Capsule Network #
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](LICENSE)

### Description
This repository contains the Python code for my Master's Thesis, which consists of the implementation of a capsule network for multi-label classification. The classification has been carried out on a set of images of dumpster and their surroundings provided by Ecoembes.
______
This code is a modification of the one presented in the paper [_Visual classification of dumpsters with capsule networks_](https://link.springer.com/content/pdf/10.1007/s11042-022-12899-9.pdf)

#### Abstract of the paper

Garbage management is an essential task in the everyday life of a city. In many countries, dumpsters are owned and deployed by the public administration. An updated what-and-where list is in the core of the decision making process when it comes to remove or renew them as well as it may give extra information to other analytics in a smart city context. In this paper we present a Capsule Network that attains a 95.35% of accuracy in recognition over the largest dataset of dumpsters available nowadays.

-----

### Run the experiment
1. Clone the repository.
```
git clone git@github.com:elenagiraldo3/TFMCapsNet.git
cd TFMCapsNet
```
2. Install ``PyTorch``, ``cuda`` and ``torchvision``. Install other needed packages.
```shell
pip install -r requirements.txt
```

3. To run the experiment you will need to separate the CSV file in train, validation and test files. You can do that with 
the ``train_test_split`` script.
```sh
python  train_test_split.py --dataset punto --path /path/to/the/file.csv --testSize 0.2 --validationSize 0.2 
```

4. Once you have the three datasets, you can start training:
```sh
python Main.py --dataset 'punto' --data_folder path/to/the/csvs/folder/ --epochs 50 --batch_size 128 
```
