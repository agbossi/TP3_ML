import os
import pickle
import pandas as pd
import svm_utils as utl
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt


def plot_color_map(predictions, image):
    # create discrete colormap
    data = np.reshape(predictions, image.shape)
    # data = [[1, 2, 3, 3, 2, 1], [3, 3, 2, 1, 1, 2], [1, 1, 1, 1, 1, 1]]
    cmap = colors.ListedColormap(['blue', 'green', 'brown'])
    bounds = [1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)

    fig = plt.figure()
    plt.imshow(image)


# load the model from disk
C = 0.0001
kernel = 'rbf'
run = str(C) + '_' + kernel
model_cielo_pasto = pickle.load(open('model_cielo_pasto' + run + '.sav', 'rb'))
model_cielo_vaca = pickle.load(open('model_cielo_vaca' + run + '.sav', 'rb'))
model_pasto_vaca = pickle.load(open('model_pasto_vaca' + run + '.sav', 'rb'))

path = os.path.abspath('test_set')
test_set = pd.read_csv(path)
# path = os.path.abspath('images/cow.jpg')
# image = plt.imread(path)

prediction_c_p = model_cielo_pasto.predict(test_set[['R', 'G', 'B']])
prediction_c_v = model_cielo_vaca.predict(test_set[['R', 'G', 'B']])
prediction_p_v = model_pasto_vaca.predict(test_set[['R', 'G', 'B']])

# Tabla que junta todos los valores de las predicciones de cada hiperplano para el testing
predicc = np.column_stack([prediction_c_p, prediction_c_v, prediction_p_v])
# predictions_df = pd.DataFrame(predicc, columns=['cielo_pasto', 'cielo_vaca', 'pasto_vaca'])
confusion_matrix, predictions = utl.get_confusion_matrix(test_set, predicc)
confusion_matrix.print_confusion_matrix()
# plot_color_map(predictions, image)