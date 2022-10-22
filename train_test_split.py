from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Path to dataset')
parser.add_argument('--testSize', type=float, default=0.20, help='Size of test dataset')
parser.add_argument('--validationSize', type=float, default=0.20, help='Size of validation dataset')

args = parser.parse_args()
# split the data into train, validation and test set
folder = os.path.dirname(args.dataset)
data = pd.read_csv(args.dataset, delimiter=';', encoding='ISO-8859-1')
# full, _ = train_test_split(data, test_size=0.40, random_state=0)
train, test = train_test_split(data, test_size=args.testSize, random_state=0)
train, validation = train_test_split(train, test_size=args.validationSize, random_state=0)
# save the data
train.to_csv(f'{folder}/train.csv', index=False)
validation.to_csv(f'{folder}/validation.csv', index=False)
test.to_csv(f'{folder}/test.csv', index=False)
