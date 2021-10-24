from sklearn.utils import shuffle


def train_test_split(data_set, training_percent):
    sets = []
    data_set = shuffle(data_set)

    training_set = data_set.iloc[0:int(len(data_set) * training_percent), :]
    testing_set = data_set.iloc[int(len(data_set) * training_percent):len(data_set), :]

    element_set = [training_set, testing_set]
    sets.append(element_set)
    return sets