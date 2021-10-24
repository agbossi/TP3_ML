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
model_cielo_pasto = pickle.load(open('model_cielo_pasto', 'rb'))
model_cielo_vaca = pickle.load(open('model_cielo_vaca', 'rb'))
model_pasto_vaca = pickle.load(open('model_pasto_vaca', 'rb'))

test_set = pd.Dataframe('test_set')
# path = os.path.abspath('images/cow.jpg')
# image = plt.imread(path)

prediction_c_p = model_cielo_pasto.predict(test_set[['R', 'G', 'B']])
prediction_c_v = model_cielo_vaca.predict(test_set[['R', 'G', 'B']])
prediction_p_v = model_pasto_vaca.predict(test_set[['R', 'G', 'B']])

# Tabla que junta todos los valores de las predicciones de cada hiperplano para el testing
Predicc = {'cielo_pasto': prediction_c_p,
           'cielo_vaca': prediction_c_v,
           'pasto_vaca': prediction_p_v}
predictions_df = pd.DataFrame(Predicc, columns=['cielo_pasto', 'cielo_vaca', 'pasto_vaca'])

confusion_matrix, predictions = utl.get_confusion_matrix(test_set, predictions_df)
confusion_matrix.print_confusion_matrix()
# plot_color_map(predictions, image)