#%% [markdown]
#Telco Customer Churn Azure ML
#https://bugra.github.io/work/notes/2014-11-22/an-introduction-to-supervised-learning-scikit-learn/
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import os

#%% [markdown]
#Azure ML libraries
import azureml.core
from azureml.core import Workspace

#%% [markdown]
#When running for the first time write config to disk
# ws = Workspace.get(name='aml', subscription_id='2974583e-8d03-498f-b505-ccd068956560',resource_group='databricks-au')
# ws.write_config(file_name="config.json")

#%% [markdown]
#Read config from file instead
ws = Workspace.from_config()

#%% [markdown]
#Setup experiemnt in AML service
from azureml.core.experiment import Experiment
experiment = Experiment(workspace=ws, name='telco-customer-churn-local')

#%% [markdown]
# Setup context 
from azureml.core import Run
from azureml.core import ScriptRunConfig

run = Run.get_context()
run.log(name="message", value="Experiment started")

#%% [markdown]
from IPython.display import Image
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import seaborn as sns

#%% [markdown]
#Read dataset from current directory
df = pd.read_csv('telco-data.csv')
#%% [markdown]
# Now we will import pandas to read our data from a CSV file and manipulate it for further use. We will also use numpy to convert out data into a format suitable to feed our classification model. We'll use seaborn and matplotlib for visualizations. We will then import Logistic Regression algorithm from sklearn. This algorithm will help us build our classification model. Lastly, we will use joblib available in sklearn to save our model for future use.

df.head(3)
yc = df["churn"].value_counts()
sns.barplot(yc.index, yc.values)#%% [markdown]
#Descriptive Analysis
df.describe()

#%% [markdown]
#Churn By State
df.groupby(["state", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30,10)) 

#%% [markdown]
#Churn By Area Code
df.groupby(["area code", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 

#%% [markdown]
#Churn By Customers with International plan
df.groupby(["international plan", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 

#%% [markdown]
#Churn By Customers with Voice mail plan¶
df.groupby(["voice mail plan", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 

#%% [markdown]
# Discreet value integer encoder
label_encoder = preprocessing.LabelEncoder()

#%% [markdown]
# State is string and we want discreet integer values
df['state'] = label_encoder.fit_transform(df['state'])
df['international plan'] = label_encoder.fit_transform(df['international plan'])
df['voice mail plan'] = label_encoder.fit_transform(df['voice mail plan'])

y = df["churn"].replace({True:1, False:0})

#print (df['Voice mail plan'][:4])
print (df.dtypes)

df.shape
df.head()

#%% [markdown]
#Strip off Redundant cols
df.drop(["phone number","churn"], axis = 1, inplace=True)

#%% [markdown]
#Build Feature Matrix
X = df.as_matrix().astype(np.float)
X.shape

#%% [markdown]
#Standardize Feature Matrix values¶
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

#%% [markdown]
#Stratified Cross Validation - Since the Response values are not balanced
from sklearn.model_selection import StratifiedKFold
def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **params):
    stratified_k_fold = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    for train, test in stratified_k_fold.split(X, y):
        # Fit
        clf = clf_class(**params)
        clf.fit(X[train], y[train])
        # Probabilistic prediction 
        y_pred[test] = clf.predict(X[test])
    return y_pred

#%% [markdown]
#Build Models and Train
print('Gradient Boosting Classifier:  {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
print('Support vector machine(SVM):   {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, svm.SVC))))
print('Random Forest Classifier:      {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print('K Nearest Neighbor Classifier: {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))
print('Logistic Regression:           {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, linear_model.LogisticRegression))))


#%% [markdown]
#Confusion Matrices for various models

grad_ens_conf_matrix      = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))
sns.heatmap(grad_ens_conf_matrix, annot=True,  fmt='');
title = 'Gradient Boosting'
plt.title(title);

#%% [markdown]
svm_svc_conf_matrix       = metrics.confusion_matrix(y, stratified_cv(X, y, svm.SVC))
sns.heatmap(svm_svc_conf_matrix, annot=True,  fmt='');
title = 'SVM'
plt.title(title);

plt.savefig('plt.png')
run.log_image("SVM", path ="plt.png")

#%% [markdown]

random_forest_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.RandomForestClassifier))
sns.heatmap(random_forest_conf_matrix, annot=True,  fmt='');
title = 'Random Forest'
plt.title(title);

plt.savefig('plt.png')
run.log_image("Random Forest", path ="plt.png")

#%% [markdown]
k_neighbors_conf_matrix   = metrics.confusion_matrix(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))
sns.heatmap(k_neighbors_conf_matrix, annot=True,  fmt='');
title = 'KNN'
plt.title(title);

#%% [markdown]
logistic_reg_conf_matrix  = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.LogisticRegression))
sns.heatmap(logistic_reg_conf_matrix, annot=True,  fmt='');
title = 'Logistic Regression'
plt.title(title);

plt.savefig('plt.png')
run.log_image("Image", path ="plt.png")

#%% [markdown]
#Classification report 

print('Gradient Boosting Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
print('Support vector machine(SVM):\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, svm.SVC))))
print('Random Forest Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print('K Nearest Neighbor Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))
print('Logistic Regression:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, linear_model.LogisticRegression))))

#%% [markdown]
#Final Model Selection

gbc = ensemble.GradientBoostingClassifier()
gbc.fit(X, y)


#%% [markdown]
# Get Feature Importance from the classifier
feature_importance = gbc.feature_importances_
print (gbc.feature_importances_)
feat_importances = pd.Series(gbc.feature_importances_, index=df.columns)
feat_importances = feat_importances.nlargest(19)
plt = feat_importances.plot(kind='barh' , figsize=(10,10)) 

#%% [markdown]
#Save model to disk
from sklearn.externals import joblib
joblib.dump(value=gbc, filename="churn-model.pkl")


#%% [markdown]
from azureml.core.model import Model
model = Model.register(workspace=ws, model_path="churn-model.pkl", model_name="telco-churn-model-test")


#%%
#Complete experiment
run.complete()
#%%
