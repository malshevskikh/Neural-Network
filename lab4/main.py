import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix

from sklearn.model_selection import GridSearchCV

df = pd.read_csv('train.csv')
print(df)

print(df['price_range'].value_counts())

x = df.drop('price_range', axis=1)
y = df['price_range']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

sc=StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)

mlp_clf = MLPClassifier(hidden_layer_sizes=(150,100,50),
                        max_iter = 300, activation = 'relu',
                        solver = 'adam')

mlp_clf.fit(trainX_scaled, trainY)

y_pred = mlp_clf.predict(testX_scaled)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

cm = confusion_matrix(testY, y_pred)
labels = np.unique(testY)
cdf = pd.DataFrame(cm, index=labels, columns=labels)
print("Матрица ошибок:")
print()
print(cdf)
print()
print("---------------------------------------")

print(classification_report(testY, y_pred))

plt.plot(mlp_clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


param_grid = {
    'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
    'max_iter': [50, 100, 150],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}


grid = GridSearchCV(mlp_clf, param_grid, n_jobs=-1, cv=5)
grid.fit(trainX_scaled, trainY)

print(grid.best_params_)

grid_predictions = grid.predict(testX_scaled)

print('Accuracy: ', accuracy_score(testY, grid_predictions))