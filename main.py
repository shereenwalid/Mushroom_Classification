# Libraries Used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


#?????????
from sklearn.utils.validation import check_is_fitted
from IPython.display import VimeoVideo


#Models
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

#                                     ***DATA ANALYSIS***

pd.set_option('display.max_columns',None)

df=pd.read_csv('mushrooms 2.csv')
# print(df)

#count of columns and rows
# print(df.shape)
#
# print(df.info())
# print(df.describe())

import plotly.express as px

data = pd.DataFrame(df)

# Create a bar chart with Plotly Express
fig = px.bar(data, x='cap-shape', y='cap-surface', color='class', title='Fruit Quantity',color_continuous_scale='edge')
fig.show()

#Graph toxicity based on feature value
fig, axes = plt.subplots(nrows=7,ncols=3, figsize=(20,50),sharey=True)
idx = 0
#skip label
# for col in df.columns[1:]:
#     sns.countplot(data=df,x=col, hue='class' ,ax=axes[idx//3][idx%3], palette='ch:s=0.25,rot=-0.25')
#     idx +=1
# plt.show()
# There are some types that are most likely safe to eat, which are:

# mushroom with almond odor or no-odor.
# mushroom with bruises.
# mushroom with gill attached
# mushroom with broad gill size
# brown, purple and white colored mushroom
# mushroom with club, equal or rooted stalk-root
# when stalk surface (above or below ring) is fibrous or smooth
# when stalk color (above or below ring) is red, white, green or orange
# mushroom with brown or orange veil-color (which is rare)
# mushroom with ring number two
# mushroom with ring type pendant or flaring
# any spore-print-color are often edible except (chocolate, green and white)
# the abundant, clustered or numerous population are often safe to eat
# mushroom habitat: waste or meadows
# - the unknown stalk root type is most likely poisonous (good to be in a separate category) otherwise we could try to impute it to the other categories





#                                            ***DATA PREPROCESSING***

#only one unique value "biased"
df.drop(columns='veil-type', inplace=True)

#null value in stalk root
df = df.replace("?", np.nan)

#stalk-root has  "2480" null values
# print(df.isnull().sum())

#replace null by most frequent
#not necessary to write b
# imputer = SimpleImputer(missing_values=np.NaN,strategy ="most_frequent" ,fill_value='b')
imputer = SimpleImputer(missing_values=np.NaN,strategy ="most_frequent")

df['stalk-root']=imputer.fit_transform(df['stalk-root'].values.reshape(-1, 1))

#null successfully replace
# print(df.isnull().sum().sum())

#Graph Edible vs Poisonous
# df['class'].value_counts().plot(kind='bar',title='Edible vs. Poisonous');
# df['class'].value_counts()


#convert categorical values to numerical
le = LabelEncoder()
def label_encoded(labels):
    le.fit(labels)
    print(labels.name,le.classes_)
    return le.transform(labels)
print("data after encoding \n")
for col in df.columns:
    df[str(col)] = label_encoded(df[str(col)])


#Split into data and lablel
x = df.drop('class',axis=1)
y = df['class']

#scale data prevent being biased
scaler = StandardScaler()
scaler.fit_transform(x)
x = pd.DataFrame(x)

#Data Before Feature Selection
x_train_beforselection,x_test_beforselection,y_train_beforselection,y_test_beforselection\
    =train_test_split(df,y,test_size=0.2,random_state=42)

#Data After Feature Selection
pca = PCA(n_components=7)
pca_fit = pca.fit_transform(x)

#TEAM QUESTION WHATS THAT??
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)

x_train_afterselection,x_test_afterselection,y_train_afterselection,y_test_afterselection\
    =train_test_split(pca_fit,y,test_size=0.2,random_state=42)

print("PCA FEATURES",pd.DataFrame(pca_fit))


#                                      ***BEFORE SELECTION***

#BEGIN  DECISION TREE
clf = DecisionTreeClassifier(criterion='entropy', max_depth=11, max_features=5,min_samples_leaf=5,random_state=1)
clf.fit(x_train_beforselection,y_train_beforselection)



depth_hyperparams = range(1, 16)
training_acc = []
validation_acc = []
for d in depth_hyperparams:
    model_dt =make_pipeline(
    DecisionTreeClassifier(max_depth=d,random_state=42)
    )
    model_dt.fit(x_train_beforselection, y_train_beforselection)
    training_acc.append(model_dt.score(x_train_beforselection,y_train_beforselection))
    validation_acc.append(model_dt.score(x_test_beforselection,y_test_beforselection))


plt.plot(depth_hyperparams,training_acc,label='training best depth')
plt.plot(depth_hyperparams,validation_acc,label='validation best depth')

clf.score(x_train_beforselection,y_train_beforselection)

print(plot_tree(clf))

y_pred_beforselection = clf.predict(x_test_beforselection)

print(y_pred_beforselection)

#100% Accuracy
print(accuracy_score(y_test_beforselection,y_pred_beforselection))

#End Decicion Tree



#BEGIN KNN


#n_neighbors num of k get by square root rows
#p num of options poisonous or not
knn = KNeighborsClassifier(n_neighbors = 91,p = 2,metric='euclidean')

#train model with the training set and training vector
knn.fit(x_train_beforselection,y_train_beforselection)

#test model with the testing set
y_pred_beforselection= knn.predict(x_test_beforselection)


#confusion matrix
KNN_confusionmatrix = confusion_matrix(y_test_beforselection,y_pred_beforselection)


# diagonal is true predicted
print(KNN_confusionmatrix)

plt.figure(figsize=(7,5))
plt.title('Confusion Matrix for KNN Classifier')
sns.heatmap(KNN_confusionmatrix , annot=True,xticklabels=["Edible", "Poisonous"],
           yticklabels=["Edible", "Poisonous"])
plt.xlabel('Predicted')
plt.ylabel('Correct Label')

# e because all features used
#diagonal is the correctly predicted
plt.show()

#classification report ( accuracy / fscore / recall / precision )
#support shows the number of instances in each class in the dataset
KNN_Report_BeforeSelection = classification_report(y_test_beforselection,y_pred_beforselection)
print(KNN_Report_BeforeSelection)

#END KNN

# BEGIN BAYESIAN CLASSIFIER
bayesian_classifier = GaussianNB()
bayesian_classifier.fit(x_train_beforselection,y_train_beforselection)

bayesian_classifier.predict(x_test_beforselection)


# Heat Map the more the diagonal is colourful the model is great
bayesian_classifier_confusion_matrix = confusion_matrix(y_test_beforselection,y_pred_beforselection)

plt.figure(figsize=(7,5))
plt.title('Confusion Matrix for Bayesian Classifier')
sns.heatmap(bayesian_classifier_confusion_matrix , annot=True, xticklabels=["Edible", "Poisonous"],
           yticklabels=["Edible", "Poisonous"])
plt.xlabel('Predicted')
plt.ylabel('Correct Label')
plt.show()

bayesian_classifier_classification_report = classification_report(y_test_beforselection,y_pred_beforselection)




# END BAYESIAN CLASSIFIER
#
# #                                      **AFTER SELECTION**
#
# # BEGIN DECISION TREE
# clf = DecisionTreeClassifier(criterion='entropy', max_depth=11, max_features=5,min_samples_leaf=5,random_state=1)
# clf.fit(x_train_afterselection,y_train_afterselection)
#
#
#
# depth_hyperparams = range(1, 16)
# training_acc = []
# validation_acc = []
# for d in depth_hyperparams:
#     model_dt =make_pipeline(
#     DecisionTreeClassifier(max_depth=d,random_state=42)
#     )
#     model_dt.fit(x_train_afterselection, y_train_afterselection)
#     training_acc.append(model_dt.score(x_train_afterselection,y_train_afterselection))
#     validation_acc.append(model_dt.score(x_test_afterselection,y_test_afterselection))
#
#
# plt.plot(depth_hyperparams,training_acc,label='training best depth')
# plt.plot(depth_hyperparams,validation_acc,label='validation best depth')
#
# clf.score(x_train_afterselection,y_train_afterselection)
#
# print(plot_tree(clf))
#
# y_pred_afterselection = clf.predict(x_test_afterselection)
#
# print(y_pred_afterselection)
#
# #96% Accuracy
# print(accuracy_score(y_test_afterselection,y_pred_afterselection))
#
# #END DECISION TREE
# # "96% after selection
#
#
#
# # LogisticRegression After Selection
#
# # model_l = make_pipeline(OneHotEncoder(),LogisticRegression(max_iter=1000))
# #
# # model_l.fit(x_train,y_train)
# # model_l.score(x_train,y_train)
# # model_l.score(x_test,y_test)
# #
#
# # Random Forest Classifier After Selection
#
# model_r= RandomForestClassifier()
# model_r.fit(x_train_afterselection, y_train_afterselection)
# model_r.predict(x_test_afterselection)
# model_r.score(x_train_afterselection,y_train_afterselection)
# model_r.score(x_test_afterselection,y_test_afterselection)




# # BEGIN KNN

# model_k=KNeighborsClassifier()
# model_k.fit(x_train,y_train)
# model_k.predict(x_test)
# model_k.score(x_train,y_train)
# model_k.score(x_test,y_test)
#
#
# #n_neighbors num of k get by square root rows
# #p num of options poisonous or not
# knn = KNeighborsClassifier(n_neighbors = 91,p = 2,metric='euclidean')
#
# #train model with the training set and training vector
# knn.fit(x_train_afterselection,y_train_afterselection)
#
# #test model with the testing set
# y_pred_afterselection = knn.predict(x_test_afterselection)
#
#
# #compare the real output with the predicted one to get accuracy of the model
# KNN_accuracy = accuracy_score(y_test_afterselection,y_pred_afterselection)
# KNN_f1score = f1_score(y_test_afterselection,y_pred_afterselection)
# KNN_confusionmatrix = confusion_matrix(y_test_afterselection,y_pred_afterselection)
# print(KNN_accuracy)
# print(KNN_f1score)
#
#
# plt.figure(figsize=(7,5))
# plt.title('Confusion Matrix for KNN Classifier')
#
# sns.heatmap(KNN_confusionmatrix , annot=True,xticklabels=["Edible", "Poisonous"],
#            yticklabels=["Edible", "Poisonous"])
# plt.xlabel('Predicted')
# plt.ylabel('Correct Label')
#
# # e because all features used
# #diagonal is the correctly predicted
# plt.show()
#
# KNN_Report_AfterSelection = classification_report(y_test_beforselection,y_pred_beforselection)
# print(KNN_Report_AfterSelection)

# #END KNN