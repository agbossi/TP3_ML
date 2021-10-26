from sklearn import svm
import generate_points
import utils

tp3_1 = generate_points.load_points("points/TP3-1.txt")

X = utils.get_X_from_points(tp3_1)
y = utils.get_Y_from_points(tp3_1)


clf = svm.SVC(kernel='linear')
clf.fit(X, y)

