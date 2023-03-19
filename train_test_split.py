import numpy as np
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
# columns = ["FOTO_CONTENEDOR", "ENVASES", "ORGANICA", "PAPEL_CARTON", "RESTOS", "VIDRIO", "SOTERRADO", "ESTADO_GRAFITI",
#            "ESTADO_QUEMADO"]
columns = ['FOTO_PUNTO', 'ENT_BOLSAS_EELL', 'ENT_PAPEL_CARTON_DOMESTICO', 'ENT_PAPEL_CARTON_INDUSTRIAL',
                       'ENT_VIDRIO', 'ENT_BOLSAS_RESTO', 'ENT_PODAS', 'ENT_ESCOMBROS', 'ENT_OBJETOS_VOLUMINOSOS',
                       'ENT_OTROS']
data = pd.read_csv(args.dataset, delimiter=';', encoding='ISO-8859-1', usecols=columns)
data.drop_duplicates()

X = data['FOTO_PUNTO']
y = data.drop(columns=['FOTO_PUNTO'])
# X_full, _, y_full, _ = train_test_split(X, y, test_size=0.8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.testSize)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.validationSize)

# save the data
train_df = pd.concat([X_train, y_train], axis=1)
train_df.to_csv(f'{folder}/trainfull_punto.csv', index=False)

val_df = pd.concat([X_val, y_val], axis=1)
val_df.to_csv(f'{folder}/validationfull_punto.csv', index=False)

test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv(f'{folder}/testfull_punto.csv', index=False)

for column in y.columns:
    y_bueno = np.count_nonzero(y[column])
    y_test_bueno = int(y_bueno * 0.2)
    y_val_bueno = int((y_bueno-y_test_bueno)*0.2)
    y_train_bueno = int(y_bueno-y_val_bueno-y_test_bueno)
    print(column)
    print("y_train=", np.count_nonzero(y_train[column]), " y tendría que ser ", y_train_bueno)
    print("y_val=", np.count_nonzero(y_val[column]), " y tendría que ser ", y_val_bueno)
    print("y_test=", np.count_nonzero(y_test[column]), " y tendría que ser ", y_test_bueno)

