# Packages needed
```
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB,  BernoulliNB, MultinomialNB
import seaborn as sns
import collections
from collections import Counter
from matplotlib.colors import ListedColormap
from numpy import where
from matplotlib import pyplot
from sklearn.tree import export_graphviz
from io import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import mlxtend.plotting
from mlxtend.plotting import plot_decision_regions
from sklearn.cluster import KMeans
```

### These models will predict goals based on the data we cleaned and created. Body part, shot position, assist position and time of game. Import all 5 countries full_shots csv files. Eliminate all outliers of shots that were on the shooting teams side of the field to prevent misrepresentaing data. Impute the average assist position for all goals that were unassisted (having 0s in the assist positions)
```
england = england.loc[(england['x start'] >= 50)]
england['assist x start'] = england['assist x start'].replace({0: 81})
```

### Choose random country and team from the country to make a train test split for finding goal predictions for specific teams, or use the whole country. In these models I will be using the entire country of England
```
X = england[['x start', 'y start','eventSec','assist y start','assist x start', 'left foot','right foot','head/body']]
y = england['goal']
liverpool = england.loc[(england['teamId'] == 1612)]
X_liverpool =liverpool[['x start', 'y start','eventSec', 'assist y start','assist x start','left foot','right foot','head/body']]
y_liverpool = liverpool['goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=94)
```

#### Using SMOTE resampling because teh number of shots to goals ratio will create a massively unbalnced data set, so resample to a 2:1 shot to goal ratio
```
smote = SMOTE(sampling_strategy=0.2)
under_sample = RandomUnderSampler(sampling_strategy=0.5)
steps = [('smote', smote),('under', under_sample)]
pipeline = Pipeline(steps = steps)
X_re, y_re = pipeline.fit_resample(X_train, y_train)
print('Resampled dataset shape %s' % Counter(y_re))

sc = StandardScaler()
X_re= sc.fit_transform(X_re)
X_test= sc.transform(X_test)

counter = Counter(y_re)
counter2 = Counter(y_train)
for label, _ in counter.items():
    row_ix = where(y_re == label)[0]
    pyplot.scatter(X_re[row_ix, 0], X_re[row_ix, 1], label=str(label))
pyplot.legend()
plt.title("Training Shots and Goals\n After Resampling", fontsize =15)

pyplot.show()
```
![image](https://user-images.githubusercontent.com/70713627/210659221-a0f120da-c830-4d29-95be-2554da474790.png)



```
sns.scatterplot(data= england, x="x start", y="y start", hue="goal")
plt.title("Training Shots and Goals\n Before Resampling", fontsize =15)
```
![image](https://user-images.githubusercontent.com/70713627/210659310-8f5f3809-a5fb-4ac8-a5c7-9f395e70c8e9.png)

# Fitting and evaluating ML Models

## Naive Bayes
```
nb_clf = GaussianNB()
nb_clf.fit(X_re, y_re)
y_pred = nb_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

f1_score(y_test, y_pred)

xlabels = ['Predicted No Goal', 'Predicted Goal']
ylabels = ['Actual No Goal', 'Actual Goal']

sns.heatmap(cm,annot=True, fmt = 'g',cmap="GnBu", xticklabels = xlabels, yticklabels = ylabels, cbar=False)
plt.title("Naive Bayes Model Performance on England", fontsize =15)
```

![image](https://user-images.githubusercontent.com/70713627/210661237-73616f35-4c9c-4d52-9cde-304ff360d633.png)

## K Nearest Neighbors

```
knn = KNeighborsClassifier(n_neighbors=15, p = 2, leaf_size = 2)
knn.fit(X_re, y_re)
y_pred = knn.predict(X_test)

f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True, fmt = 'g',cmap="GnBu", xticklabels = xlabels, yticklabels = ylabels, cbar=False)
plt.title("KNN Model Performance on England", fontsize =15)
```
![image](https://user-images.githubusercontent.com/70713627/210662006-39282115-06b2-4958-86ba-f1d6fd23a677.png)

## Choosing number of neighbors

```
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_re,y_re)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
 markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

```
![image](https://user-images.githubusercontent.com/70713627/210663523-8f9c2fbc-d223-4d65-a516-4378986718ce.png)


## Random Forest 

```
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=15, max_depth = 8, criterion = 'gini', random_state=94)
rnd_clf.fit(X_re, y_re)

y_pred_rf = rnd_clf.predict(X_test)

f1_score(y_test, y_pred_rf)

cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm,annot=True, fmt = 'g',cmap="GnBu", xticklabels = xlabels, yticklabels = ylabels, cbar=False)
plt.title("Random Forest Model Performance on England", fontsize =15)
```
![image](https://user-images.githubusercontent.com/70713627/210662587-ca5aa76d-a3fa-40cb-9eb9-3e18ccb25b47.png)

## Feature Importance from Random Forest 

```
f_imp = rnd_clf.feature_importances_
dict = {'Feature':['Shot Distance','Shot Angle','Time of Game','Assist Distance','Assist Angle','Left Foot','Right Foot','Head/Body'],
       'Importance':[0.13048475, 0.04612654, 0.00889569, 0.34097953, 0.39967551,
       0.02146678, 0.02001951, 0.03235169]}
f_impdf= pd.DataFrame(dict)
ax = sns.barplot(data= f_impdf, x="Feature", y = 'Importance')
plt.title("Feature Importance of Random Forest", fontsize =15)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
```
![image](https://user-images.githubusercontent.com/70713627/210662789-b4937699-99ae-473e-93b4-e8aadeb8f03c.png)





