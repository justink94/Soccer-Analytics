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

#These models will predict goals based on the data we cleaned and created. Body part, shot position, assist position and time of game
#Import all 5 countries full_shots csv files
#Eliminate all outliers of shots that were on the shooting teams side of the field to prevent misrepresentaing data
#Impute the average assist position for all goals that were unassisted (having 0s in the assist positions)

england = england.loc[(england['x start'] >= 50)]
england['assist x start'] = england['assist x start'].replace({0: 81})

#Choose random country and team from the country to make a train test split for finding goal predictions for specific teams, or use the whole country

X = england[['x start', 'y start','eventSec','assist y start','assist x start', 'left foot','right foot','head/body']]
y = england['goal']
liverpool = england.loc[(england['teamId'] == 1612)]
X_liverpool =liverpool[['x start', 'y start','eventSec', 'assist y start','assist x start','left foot','right foot','head/body']]
y_liverpool = liverpool['goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=94)

#Using SMOTE resampling because teh number of shots to goals ratio will create a massively unbalnced data set, so resample to a 2:1 shot to goal ratio
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

sns.scatterplot(data= england, x="x start", y="y start", hue="goal")
plt.title("Training Shots and Goals\n Before Resampling", fontsize =15)
