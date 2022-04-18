import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
# Let us use RFE to check required features and remove multicolearity
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
# Import label encoder
from sklearn import preprocessing
from imblearn.under_sampling import NeighbourhoodCleaningRule 
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# Import Dataset
heart = pd.read_csv('Heart_dataset.csv')

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
# Encode labels in column 'species'.
heart['HeartDisease']= label_encoder.fit_transform(heart['HeartDisease'])
heart['Smoking']= label_encoder.fit_transform(heart['Smoking'])
heart['AlcoholDrinking']= label_encoder.fit_transform(heart['AlcoholDrinking'])
heart['Stroke']= label_encoder.fit_transform(heart['Stroke'])
heart['DiffWalking']= label_encoder.fit_transform(heart['DiffWalking'])
heart['Sex']= label_encoder.fit_transform(heart['Sex'])
heart['AgeCategory']= label_encoder.fit_transform(heart['AgeCategory'])
heart['Race']= label_encoder.fit_transform(heart['Race'])
heart['Diabetic']= label_encoder.fit_transform(heart['Diabetic'])
heart['PhysicalActivity']= label_encoder.fit_transform(heart['PhysicalActivity'])
heart['GenHealth']= label_encoder.fit_transform(heart['GenHealth'])
heart['Asthma']= label_encoder.fit_transform(heart['Asthma'])
heart['KidneyDisease']= label_encoder.fit_transform(heart['KidneyDisease'])
heart['SkinCancer']= label_encoder.fit_transform(heart['SkinCancer'])

X, y = heart.loc[:, heart.columns != 'HeartDisease'], heart['HeartDisease']
ncr = NeighbourhoodCleaningRule(n_neighbors=20, threshold_cleaning=0.5)

X_ncr, y_ncr =ncr.fit_resample(X,y)

#Train Test Split
X_train, X_test,y_train,y_test = train_test_split(X_ncr,y_ncr,test_size=0.40,random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# XGBoost Classifier with Bagging and Boosting

# Init classifier
xgb_cl = xgb.XGBClassifier()

# The baggging ensemble classifier is initialized with:

bagging = BaggingClassifier(base_estimator=xgb_cl, n_estimators=5, max_samples=50, bootstrap=True)

# Training
bagging.fit(X_train, y_train)

# Evaluating
st.write(f"Train score: {bagging.score(X_train, y_train)}")
st.write(f"Test score: {bagging.score(X_test, y_test)}")

# Fit
xgb_cl.fit(X_train, y_train)

# Predict
preds = xgb_cl.predict(X_test)

# Score
accuracy_score(y_test, preds)
