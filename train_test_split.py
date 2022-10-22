from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
from skimage import io

# split the data into train, validation and test set
data = pd.read_csv("C:/Users/elena/Documents/TFM/AA_TAG_IMAGENES/datosContenedorGO.csv",
                   delimiter=';', encoding='ISO-8859-1')
# full, _ = train_test_split(data, test_size=0.40, random_state=0)
train, test = train_test_split(data, test_size=0.20, random_state=0)
train, validation = train_test_split(train, test_size=0.20, random_state=0)
# save the data
train.to_csv('C:/Users/elena/Documents/TFM/AA_TAG_IMAGENES/train.csv', index=False)
validation.to_csv('C:/Users/elena/Documents/TFM/AA_TAG_IMAGENES/validation.csv', index=False)
test.to_csv('C:/Users/elena/Documents/TFM/AA_TAG_IMAGENES/test.csv', index=False)
