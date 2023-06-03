import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path to dataset')
parser.add_argument('--testSize', type=float, default=0.20, help='Size of test dataset')
parser.add_argument('--validationSize', type=float, default=0.20, help='Size of validation dataset')
parser.add_argument('--dataset', type=str, default='punto', help='Dataset to use, punto or contenedor')

args = parser.parse_args()
# split the data into train, validation and test set
folder = os.path.dirname(args.path)
dataset = args.dataset

if dataset == 'contenedor':
    columns = ["FOTO_CONTENEDOR", "ENVASES", "ORGANICA", "PAPEL_CARTON", "RESTOS", "VIDRIO", "SOTERRADO",
               "ESTADO_GRAFITI", "ESTADO_QUEMADO"]
else:
    columns = ['FOTO_PUNTO', 'ENT_BOLSAS_EELL', 'ENT_PAPEL_CARTON_DOMESTICO', 'ENT_PAPEL_CARTON_INDUSTRIAL',
               'ENT_VIDRIO', 'ENT_BOLSAS_RESTO', 'ENT_PODAS', 'ENT_ESCOMBROS', 'ENT_OBJETOS_VOLUMINOSOS',
               'ENT_OTROS', 'DESBORDE_RESTO', 'DESBORDE_EELL', 'DESBORDE_PC', 'DESBORDE_VIDRIO', 'ESTABL_COMERCIOS',
               'ESTABL_HORECA', 'ESTABL_OBRAS', 'ESTABL_CENTROPUBLICO']

data = pd.read_csv(args.path, delimiter=';', encoding='ISO-8859-1', usecols=columns)
data.drop_duplicates()

X = data[f'FOTO_{dataset.upper()}']
y = data.drop(columns=[f'FOTO_{dataset.upper()}'])
X_full, _, y_full, _ = train_test_split(X, y, test_size=0.99)
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=args.testSize)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.validationSize)

# save the data
train_df = pd.concat([X_train, y_train], axis=1)
train_df.to_csv(f'{folder}/train_{dataset}.csv', index=False)

val_df = pd.concat([X_val, y_val], axis=1)
val_df.to_csv(f'{folder}/validation_{dataset}.csv', index=False)

test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv(f'{folder}/test_{dataset}.csv', index=False)