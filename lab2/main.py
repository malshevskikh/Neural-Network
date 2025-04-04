
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv('train.csv')
test = pd.read_csv('train.csv')
#print(train_df)
#train_df = train_df.drop('price_range', axis=1)
#train_df = pd.read_csv('train.csv')
print(train)


f = train.drop(['price_range'], axis=1)
print(f.shape)
t = train['price_range']
f_train, f_test, t_train, t_test = train_test_split(f, t, test_size=0.2)
print("----------------------------------")
#print(f_test.shape)
#print(f_train.shape)
#print(t_test.shape)
#print(t_train.shape)

k_values = [i for i in range (1,31)]
scores = []
cvs = []

print("----------------К-ближайших соседей---------------")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    best_cv = 1
    scores_temp = []
    cvvals = list(range(2, 11))
    for i in cvvals:
      # score = cross_val_score(knn, features, target, cv=i)
      score = cross_val_score(knn, f_train, t_train, cv=i)
      scores_temp.append(np.mean(score))
    #print("scores_temp", scores_temp)
    fig = plt.figure()
    plt.plot(cvvals, scores_temp)
    plt.xlabel("CV Values")
    plt.ylabel("Accuracy Score")
    plt.savefig("KNN/Accuracy from CV with K=" + str(k))
    plt.close()

    best_index = np.argmax(scores_temp)
    cvs.append(best_index)
    scores.append(scores_temp)

print("scores 1", scores)
print("cvs", cvs)
transposed = np.array(scores).T.tolist()
for i, v in enumerate(transposed):
    transposed[i] = sum(v)
print("transposed", transposed)
cv_index = np.argmax(transposed)
print("cv_index", cv_index)
for i, v in enumerate(scores):
    scores[i] = v[cv_index]

print("scores 2", scores)
print("k values", k_values)
fig = plt.figure()
plt.plot(k_values, scores)
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.savefig("KNearestNeighbours - Acuracy Graph")
plt.close()

best_index = np.argmax(scores)
best_k = k_values[best_index]
print("best k", best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(f_train, t_train)

t_pred = knn.predict(f_test)

cm = confusion_matrix(t_test, t_pred)
labels = np.unique(t_test)
cdf = pd.DataFrame(cm, index=labels, columns=labels)
print("Матрица ошибок для К ближайших соседей:")
print()
print(cdf)
print()
print("---------------------------------------")


#Checking performance our model with classification report.
print(classification_report(t_test, t_pred))

#Checking performance our model with ROC Score.
print("Accuracy: ", accuracy_score(t_test, t_pred))
print("---------------------------------------")

np.savetxt("knn-predicted.csv", t_pred, delimiter=', ', fmt ='% s')

print("----------------Дерево принятия решений----------------")

params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 6, 8, 10],
    'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    'splitter': ['best', 'random']
}

scores = []
params_list = []
for i in list(range(2, 11)):
    clf = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, cv=i)
    # clf.fit(features, target)
    clf.fit(f_train, t_train)
    params_list.append(clf.best_params_)
    scores.append(clf.best_score_)

fig = plt.figure()
plt.plot(list(range(2, 11)), scores_temp)
plt.xlabel("CV Values")
plt.ylabel("Accuracy Score")
plt.savefig("Tree/Accuracy from CV")
plt.close()


best_index = np.argmax(scores)
best_params = params_list[best_index]
print("best params:", best_params)

scores = []
print("max depth:", params['max_depth'])
for i in params['max_depth']:
    dtc = DecisionTreeClassifier(criterion=best_params['criterion'], \
                                 max_depth=i, \
                                 max_features=best_params['max_features'], \
                                 splitter=best_params['splitter']
                                 )
    dtc.fit(f_train, t_train)
    t_pred = dtc.predict(f_test)
    scores.append(accuracy_score(t_test, t_pred))

fig = plt.figure()
plt.plot(params['max_depth'], scores)
plt.xlabel("Max Depth Values")
plt.ylabel("Accuracy Score")
plt.savefig("Decision Tree - Accuracy Graph")
plt.close()

# clf = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, cv=5)

# clf.fit(f_train, t_train)

dtc = DecisionTreeClassifier(criterion=best_params['criterion'], max_depth=best_params['max_depth'], max_features=best_params['max_features'], splitter=best_params['splitter'])
dtc.fit(f_train, t_train)
t_pred_2 = dtc.predict(f_test)

cm = confusion_matrix(t_test, t_pred)
labels = np.unique(t_test)
cdf = pd.DataFrame(cm, index=labels, columns=labels)
print("Матрица ошибок для дерева принятия решений:")
print()
print(cdf)
print()
print("---------------------------------------")

# Checking performance our model with classification report.
print(classification_report(t_test, t_pred_2))

# Checking performance our model with ROC Score.
print("Accuracy: ", accuracy_score(t_test, t_pred_2))
print("---------------------------------------")

np.savetxt("decision-tree-predicted.csv", t_pred_2, delimiter=', ', fmt='% s')
#print("^^^^^^^^^^^^^^^^  ", t_pred)
t_pred = np.insert(t_pred, 0, 1111)
t_pred_2 = np.insert(t_pred_2, 0, 2222)
t_test = np.insert(np.array(t_test), 0, 3333)

total = [t_pred, t_pred_2, t_test]

total = np.array(total).T.tolist()

np.savetxt("full-prediction.csv", total, delimiter=', ', fmt='% s')



'''
print("----------------К-ближайших соседей---------------")

knn = KNeighborsClassifier(n_neighbors= 5)
scores = cross_val_score(knn, X, y , cv = 10, scoring= 'accuracy')
print(scores)
print(scores.mean())

k_range = range(1, 41)
k_scores = []
for k in k_range:
  knn = KNeighborsClassifier(n_neighbors= k)
  scores = cross_val_score(knn, X, y , cv = 5, scoring= 'accuracy')
  k_scores.append(scores.mean())

print(k_scores)

sn.lineplot(x = k_range, y = k_scores)
plt.show()


print("----------------Метод дерева принятия решений---------------")
X = train.drop('price_range', axis=1)
y = train['price_range']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("-------------------------")

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

k_range = range(1, 41)
k_scores = []
for k in k_range:
  clf = tree.DecisionTreeClassifier(max_depth = k)
  scores = cross_val_score(estimator=clf, X=X, y=y, cv=7, n_jobs=4)
  k_scores.append((k, scores.mean()))

print(k_scores)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''

'''
#----------------Метод k-соседей---------------
predictors = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height' , 'px_width' , 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']
outcome = 'price_range'

new_record = train_df.loc[0:0, predictors]
X = train_df.loc[1:, predictors]
y = train_df.loc[1:, outcome]

kNN = KNeighborsClassifier(n_neighbors=20)
kNN.fit(X, y)
kNN.predict(new_record)
print(kNN.predict_proba(new_record))

nbrs = kNN.kneighbors(new_record)
print(nbrs)

nbr_df = pd.DataFrame({'battery_power': X.iloc[nbrs[1][0], 0],
                         'blue': X.iloc[nbrs[1][0], 1], 'clock_speed': X.iloc[nbrs[1][0], 2],
                        'dual_sim': X.iloc[nbrs[1][0], 3], 'fc': X.iloc[nbrs[1][0], 4],
                       'four_g': X.iloc[nbrs[1][0], 5], 'int_memory': X.iloc[nbrs[1][0], 6],
                       'm_dep': X.iloc[nbrs[1][0], 7], 'mobile_wt': X.iloc[nbrs[1][0], 8],
                       'n_cores': X.iloc[nbrs[1][0], 9], 'pc': X.iloc[nbrs[1][0], 10],
                        'px_height': X.iloc[nbrs[1][0], 11], 'px_width': X.iloc[nbrs[1][0], 12],
                        'ram': X.iloc[nbrs[1][0], 13], 'sc_h': X.iloc[nbrs[1][0], 14],
                        'sc_w': X.iloc[nbrs[1][0], 15], 'talk_time': X.iloc[nbrs[1][0], 16],
                        'three_g': X.iloc[nbrs[1][0], 17], 'touch_screen': X.iloc[nbrs[1][0], 18],
                        'wifi': X.iloc[nbrs[1][0], 19],
                         'price_range': y.iloc[nbrs[1][0]]})

print(nbr_df)

#----------------Метод деревая принятия решений---------------
s = dataset.shape
print(s)
print(dataset.head())
X = dataset.drop('price_range', axis=1)
y = dataset['price_range']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("-------------------------")

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''
