import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlxtend
import warnings
warnings.filterwarnings("ignore")

from ClfAutoEDA import *

df=pd.read_csv('pulsar_stars.csv')
labels=["non-pulsar","pulsar"]
target_variable_name='target_class'

df_processed,num_features,cat_features=EDA(df,labels,
                                           target_variable_name,
                                           data_summary_figsize=(6,6),
                                           corr_matrix_figsize=(6,6),
                                           corr_matrix_annot=True,
                                           pairplt=True
                                           )

#dividing the X and the y
X=df_processed.drop([target_variable_name], axis=1)
y=df_processed[target_variable_name]

#RadViz plot
from yellowbrick.features import RadViz
visualizer = visualizer = RadViz(classes=labels, features=X.columns.tolist(),size = (800,300))
visualizer.fit(X, y)      
visualizer.transform(X)  
visualizer.show()

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)




from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV


logreg=LogisticRegression()
SVM=SVC()
knn=KNeighborsClassifier()
gnb=GaussianNB()
etree=ExtraTreesClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=42)



scaler=StandardScaler()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()

features=X_train.columns.tolist()





X_train_scaled=scaler.fit_transform(X_train) 
X_test_scaled=scaler.fit_transform(X_test) 






#feature selection by feature importance
start_time = timeit.default_timer()
mod=etree
# fit the model
mod.fit(X_train_scaled, y_train)
# get importance
importance = mod.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
df_importance=pd.DataFrame({'importance':importance},index=features)
df_importance.plot(kind='barh')
#plt.bar([x for x in range(len(importance))], importance)
elapsed = timeit.default_timer() - start_time
print('Execution Time for feature selection: %.2f minutes'%(elapsed/60))

feature_imp=list(zip(features,importance))
feature_sort=sorted(feature_imp, key = lambda x: x[1]) 
n_top_features=8
top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

X_train_sfs=X_train[top_features]
X_test_sfs=X_test[top_features]

X_train_sfs_scaled=scaler.fit_transform(X_train_sfs)
X_test_sfs_scaled=scaler.fit_transform(X_test_sfs)


models=[knn,rforest,etree,SVM]
param_distributions=[{'n_neighbors':[5,10,15]},{'criterion':['gini', 'entropy'],'n_estimators':[100,200,300]},{'criterion':['gini', 'entropy'],'n_estimators':[100,200,300]},{'kernel':['rbf','linear'],'C':[0.1,1,10],'gamma':[0.1,0.01,0.001]}]

for model in models:
    rand=RandomizedSearchCV(model,param_distributions=param_distributions[models.index(model)],cv=3,scoring='accuracy', n_jobs=-1, random_state=42,verbose=10)
    rand.fit(X_train_sfs_scaled,y_train)
    print(rand.best_params_,rand.best_score_) 
    

 

    
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC
from sklearn.metrics import  classification_report

def yellowbrick_visualizations(model,classes,X_tr,y_tr,X_te,y_te):
    visualizer=ConfusionMatrix(model,classes=classes)
    visualizer.fit(X_tr,y_tr)
    visualizer.score(X_te,y_te)
    visualizer.show()
    
    visualizer = ClassificationReport(model, classes=classes, support=True)
    visualizer.fit(X_tr,y_tr)
    visualizer.score(X_te,y_te)
    visualizer.show()
    
    visualizer = ROCAUC(model, classes=classes)
    visualizer.fit(X_tr,y_tr)
    visualizer.score(X_te,y_te)
    visualizer.show()
    
classes=['Non-Pulsar','Pulsar']
model=ExtraTreesClassifier(n_estimators=300,criterion='gini',random_state=42)
#model=RandomForestClassifier(n_estimators=200,criterion='gini',random_state=42)
model.fit(X_train_sfs_scaled,y_train)
y_pred = model.predict(X_test_sfs_scaled)

print(classification_report(y_test, y_pred))
yellowbrick_visualizations(model,classes,X_train_sfs_scaled,y_train,X_test_sfs_scaled,y_test)

print(np.bincount(y_pred))
print(np.bincount(y_train))

from imblearn.over_sampling import SMOTE,RandomOverSampler,BorderlineSMOTE
from imblearn.under_sampling import NearMiss,RandomUnderSampler
smt = SMOTE()
nr = NearMiss()
bsmt=BorderlineSMOTE(random_state=42)
ros=RandomOverSampler(random_state=42)
rus=RandomUnderSampler(random_state=42)
X_train_bal, y_train_bal = smt.fit_sample(X_train_sfs_scaled, y_train)
print(np.bincount(y_train_bal))


model_bal=model
model_bal.fit(X_train_bal, y_train_bal)
y_pred = model_bal.predict(X_test_sfs_scaled)
print(classification_report(y_test, y_pred))
yellowbrick_visualizations(model_bal,classes,X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)




#Plot decision region
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions as plot_dr 

def plot_classification(model,X_t,y_t):
    clf=model
    pca = PCA(n_components = 2)
    X_t2 = pca.fit_transform(X_t)
    clf.fit(X_t2,np.array(y_t))
    plot_dr(X_t2, np.array(y_t), clf=clf, legend=2)

plot_classification(model_bal,X_test_sfs_scaled,y_test)
