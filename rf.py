import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv


train_file = "../data/train.csv"
pred_file = "../data/test.csv"

train = np.loadtxt(train_file, delimiter = ',')
pred = np.loadtxt(pred_file, delimiter = ',')

x_train = train[:, 1:-1]
y_train = train[:, -1]


x_pred = pred[:, 1::]

clf = RandomForestClassifier(n_estimators=500, max_depth = 9, n_jobs = 2)
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_pred)

p_file = open("../data/predict.csv", "wb")
p_data = csv.writer(p_file)
p_data.writerow(["id", "label"])

i = 0
for value in y_pred:
	p_data.writerow([i, int(value)])
	i+=1
p_file.close()


