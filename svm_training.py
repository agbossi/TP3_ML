import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import svm_utils as utl
from Data_resamplers import train_test_split


R = 0
G = 1
B = 2
LABEL = 3

path = os.path.abspath('images/cielo.jpg')
cielo = plt.imread(path)
cielo_class = 1

path = os.path.abspath('images/pasto.jpg')
pasto = plt.imread(path)
pasto_class = 2

path = os.path.abspath('images/vaca.jpg')
vaca = plt.imread(path)
vaca_class = 3

cielo_bands = utl.split_rgb(cielo, cielo_class)
pasto_bands = utl.split_rgb(pasto, pasto_class)
vaca_bands = utl.split_rgb(vaca, vaca_class)

R_column = np.concatenate((cielo_bands[R], pasto_bands[R], vaca_bands[R]), axis=0)
G_column = np.concatenate((cielo_bands[G], pasto_bands[G], vaca_bands[G]), axis=0)
B_column = np.concatenate((cielo_bands[B], pasto_bands[B], vaca_bands[B]), axis=0)

label_column = np.concatenate((cielo_bands[LABEL], pasto_bands[LABEL], vaca_bands[LABEL]), axis=0)

# Creación de la tabla del conjunto de datos:
data = {'R': R_column, 'G': G_column, 'B': B_column, 'Clase': label_column}
df = pd.DataFrame(data, columns=['R', 'G', 'B', 'Clase'])
df = shuffle(df)

training_percent = 0.7
sets = train_test_split(df, training_percent)[0]
training = sets[0]
test = sets[1]
C = 0.0001
kernel = 'rbf'

# enfoque uno contra uno para encontrar hiperplanos que separen clases
training_cielo_pasto = training[(training['Clase'] == cielo_class) | (training['Clase'] == pasto_class)]
training_cielo_vaca = training[(training['Clase'] == cielo_class) | (training['Clase'] == 3)]
training_pasto_vaca = training[(training['Clase'] == vaca_class) | (training['Clase'] == pasto_class)]

model_cielo_pasto = utl.svm_train(training_cielo_pasto, C=C, kernel=kernel)
model_cielo_vaca = utl.svm_train(training_cielo_vaca, C=C, kernel=kernel)
model_pasto_vaca = utl.svm_train(training_pasto_vaca, C=C, kernel=kernel)

# save the model to disk
test.to_csv('test_set.csv')
pickle.dump(model_cielo_pasto, open('model_cielo_pasto' + str(C) + '_' + kernel + '.sav', 'wb'))
pickle.dump(model_cielo_vaca, open('model_cielo_vaca' + str(C) + '_' + kernel + '.sav', 'wb'))
pickle.dump(model_pasto_vaca, open('model_pasto_vaca' + str(C) + '_' + kernel + '.sav', 'wb'))
