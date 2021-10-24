import numpy as np
import pandas as pd
from matplotlib import style
from Metrics import ConfusionMatrix
style.use("ggplot")
from sklearn.svm import SVC


R = 0
G = 1
B = 2
LABEL = 3


def split_rgb(image, class_mark):
    pixel_amouunt = len(image[0, :, R]) * len(image[:, 0, R]) # 35475
    image_R = np.reshape(image[:, :, R], pixel_amouunt, 'C')
    image_G = np.reshape(image[:, :, G], pixel_amouunt, 'C')
    image_B = np.reshape(image[:, :, B], pixel_amouunt, 'C')
    image_class = np.ones(pixel_amouunt) * class_mark

    return image_R, image_G, image_B, image_class


def svm_train(training_set, kernel='rbf', C=0.0001):
    # Entrenar la primera tabla:
    SVMmodel = SVC(kernel=kernel, C=C)
    SVMmodel.fit(training_set[['R', 'G', 'B']], training_set[['Clase']].values.ravel())
    # .values will give the values in an array. (shape: (n,1)
    # .ravel will convert that array shape to (n, )
    # SVMmodel.coef_ 'me devuelvo los valores de b'
    # prediction = SVMmodel.predict(test_set[['R', 'G', 'B']])
    return SVMmodel


def most_voted(prediction):
    class_count = [0, 0, 0]
    for elem in prediction:
        # -1 porque el minimo es 1
        class_count[elem-1] += 1
    max_value = max(class_count)
    if max_value == 1:
        pred = np.random.randint(1, 4)
    else:
        pred = class_count.index(max_value) + 1
    return pred


def summarize_predictions(predictions):
    data = []
    for prediction in predictions:
        data.append(most_voted(prediction))
    return pd.DataFrame(data=data, columns=['Clase'])


def get_confusion_matrix(testing, predictions):
    confusion_matrix = ConfusionMatrix(['1, 2, 3'])
    final_predictions = summarize_predictions(predictions)
    classifications = pd.concat([testing['Clase'], final_predictions['Clase']], axis=0)
    # classifications = zip(testing['Clase'], final_predictions['Clase'])
    for classification in classifications:
        # -1 porque las clasificaciones arrancan en 1 y romperian los indices de la matriz
        confusion_matrix.add_entry(classification.iloc[0]-1, classification.iloc[1]-1)
    confusion_matrix.summarize()
    return confusion_matrix, final_predictions
