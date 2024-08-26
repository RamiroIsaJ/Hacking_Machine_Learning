import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.metrics import confusion_matrix
from unsupervised_random_forest import urf
'''
hack = pd.read_csv('features_pattern.csv')
nHack = pd.read_csv('features_pattern_.csv')
# classes = pd.read_csv('features_classes.csv')
data1 = StandardScaler().fit_transform(hack)
data2 = StandardScaler().fit_transform(nHack)
# Apply PCA
pca = PCA(n_components=2)
pcaHack = pca.fit_transform(data1)
hackDf = pd.DataFrame(data=pcaHack, columns=['PCA1', 'PCA2'])

pcaNHack = pca.fit_transform(data2)
hackNDf = pd.DataFrame(data=pcaNHack, columns=['PCA1', 'PCA2'])
# finalDf = pd.concat([principalDf, classes['Id_class']], axis=1)
targets = ['Hacking', 'No Hacking']
colors = ['r', 'b', 'g']

plt.figure()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('HACKING and NOT HACKING DATA with PCA COMPONENTS')
plt.plot(hackDf['PCA1'], hackDf['PCA2'], 'ob')
plt.plot(hackNDf['PCA1'], hackNDf['PCA2'], 'or')

plt.legend(targets)
plt.grid()
plt.show()
'''
file = pd.read_csv('features_patternT.csv')
print(file)
'''
hack = file.loc[file['Descript'] >= 0]
nhack = file.loc[file['Descript'] < 0]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(hack['Web'], hack['IP'], hack['Descript'], 'or')
ax.scatter(nhack['Web'], nhack['IP'], nhack['Descript'], 'og')
plt.show()
'''
classes = pd.read_csv('features_classes.csv')
data = StandardScaler().fit_transform(file)
print('----> Finish standard scaler process -------------')
# Apply PCA
pca = PCA(n_components=2)
dataset1 = pca.fit_transform(data)
print(dataset1)
# dataset2 = pca.fit_transform(data2)
print('----> Finish PCA process -------------')

finalDf = pd.DataFrame(data=dataset1, columns=['PCA1', 'PCA2'])
finalDf = pd.concat([finalDf, classes['Id_class']], axis=1)
targets = [0, 1]
targets_ = ['Hacking', 'No hacking']
colors = ['b', 'r', 'g']
fig = plt.figure()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('HACKING and NOT HACKING DATA with PCA COMPONENTS')


for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Id_class'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'PCA1'], finalDf.loc[indicesToKeep, 'PCA2'],  c=color)
plt.legend(targets_)
plt.grid()


print('ONE CLASS SVM')
clf = svm.OneClassSVM(nu=0.2, kernel="linear", gamma=0.1)
X_train = dataset1
clf.fit(X_train)
y_predict = clf.predict(X_train)
y_predict_n = 1 - ((y_predict + 1) // 2)
print(y_predict_n)
plt.figure()
plt.title('ONE CLASS SVM')
for target, color in zip(targets, colors):
    indicesToKeep = y_predict_n == target
    plt.scatter(dataset1[indicesToKeep, 0], dataset1[indicesToKeep, 1],  c=color)
plt.legend(targets_)
plt.grid()
labels_test = finalDf['Id_class']
plt.figure()
plt.title('Confusion Matrix - ONE CLASS SVM')
matrix = confusion_matrix(y_predict_n, labels_test)
ax = sb.heatmap(matrix/np.sum(matrix), annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=targets_, yticklabels=targets_)
ax.set_xlabel('Predicted values')
ax.set_ylabel('Test values')
plt.show()
'''
print('Isolation Forest')
X_train = dataset1
clf = IsolationForest(contamination=0.075, random_state=0)
clf.fit(X_train)
y_predict = clf.predict(X_train)
y_predict_n = 1 - ((y_predict + 1) // 2)

plt.figure()
plt.title('Isolation Forest')
for target, color in zip(targets, colors):
    indicesToKeep = y_predict_n == target
    plt.scatter(dataset1[indicesToKeep, 0], dataset1[indicesToKeep, 1],  c=color)
plt.legend(targets_)
plt.grid()

labels_test = finalDf['Id_class']
plt.figure()
plt.title('Confusion Matrix - Isolation Forest')
matrix = confusion_matrix(y_predict_n, labels_test)
ax = sb.heatmap(matrix/np.sum(matrix), annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=targets_, yticklabels=targets_)
ax.set_xlabel('Predicted values')
ax.set_ylabel('Test values')
plt.show()


print('Unsupervised Random Forest')
X_train = dataset1
n_outliers = int(len(X_train)*0.10)
clf = urf(n_trees=500, max_depth=3)
a = clf.get_anomaly_score(X_train, knn=5)
y_predict = np.ones(a.shape, dtype=int)
outliers = np.argsort(a)[::-1][:n_outliers]
y_predict[outliers] = 0
y_predict_n = 1 - ((y_predict + 1) // 2)
plt.figure()
plt.title('Unsupervised Random Forest')
for target, color in zip(targets, colors):
    indicesToKeep = y_predict_n == target
    plt.scatter(dataset1[indicesToKeep, 0], dataset1[indicesToKeep, 1],  c=color)
plt.legend(targets_)
plt.grid()

labels_test = finalDf['Id_class']
plt.figure()
plt.title('Confusion Matrix - Unsupervised Random Forest')
matrix = confusion_matrix(y_predict_n, labels_test)
ax = sb.heatmap(matrix/np.sum(matrix), annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=targets_, yticklabels=targets_)
ax.set_xlabel('Predicted values')
ax.set_ylabel('Test values')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('HACKING and NOT HACKING DATA with PCA COMPONENTS')
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Id_class'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'PCA1'], finalDf.loc[indicesToKeep, 'PCA2'],
                finalDf.loc[indicesToKeep, 'PCA3'], c=color)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.legend(targets)
plt.grid()
plt.show()
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Id_class'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'PCA1'], finalDf.loc[indicesToKeep, 'PCA2'], c=color)
plt.legend(targets)
plt.grid()
plt.show()
'''