from ast import Global
from pydoc import text
from turtle import color
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
from sklearn.model_selection import train_test_split
#Libraries for ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.linear_model import SGDClassifier

data = pd.read_csv('dataset.csv')

#Visual_1
A0=['Male','Female']
B0=[] 
C0=[]
D0=[]
for a in A0:
    x=data.loc[(data['Sex']==a),:].loc[data['HeartDisease']=='Yes'].count()['HeartDisease']
    y=data.loc[(data['Sex'] ==a),].count()['HeartDisease']
    B0+=[x/y]
    C0+=[a]
    D0+=[x]

#Visual_2
D={}
L=['Smoking','AlcoholDrinking','Stroke','DiffWalking','PhysicalActivity','Asthma','KidneyDisease','SkinCancer']
for l in L:
    D[l]=[0,0]
Z=['No','Yes']

for l in L:
    for i in Z:
        x=data.loc[(data[l]==i),:].loc[data['HeartDisease']=='Yes'].HeartDisease.count()
        y=data.loc[(data[l]==i),].count()['HeartDisease']
        r=Z.index(i)
        D[l][r]=x/y

fig = make_subplots(rows=2, cols=4, subplot_titles=L )
B=[None]*8
r=0
for i in range(1,3):
    for j in range(1,5):
        B[r]=(i,j)
        r+=1
r=0
for l in L:
    fig.add_trace(go.Bar(x=['No','Yes'], y=D[l],name=l), row=B[r][0], col=B[r][1])
    r+=1

#Visual_3
A=list(data.groupby('Diabetic').groups.keys())
B=[]
C=[]
for p in A:
    x=data.loc[(data['Diabetic']==p),:].loc[data['HeartDisease']=='Yes'].count()['HeartDisease']
    y=data.loc[(data['Diabetic']==p),].count()['HeartDisease']
    B+=[x/y]
    C+=[y]

#Visual_4
A1=list(data.groupby('AgeCategory').groups.keys())
B1=[]
C1=[]
for p in A1:
    x=data.loc[(data['AgeCategory']==p),:].loc[data['HeartDisease']=='Yes'].count()['HeartDisease']
    y=data.loc[(data['AgeCategory']==p),].count()['HeartDisease']
    B1+=[x/y]
    C1+=[y]

#Visual_5
A2=list(data.groupby('Race').groups.keys())
B2=[]
C2=[]
for k in A2:
    x=data.loc[(data['Race']==k),:].loc[data['HeartDisease']=='Yes'].count()['HeartDisease']
    y=data.loc[(data['Race']==k),].count()['HeartDisease']
    B2+=[x/y]
    C2+=[y]

#Visual_6
A3=list(data.groupby('GenHealth').groups.keys())
B3=[]
C3=[]
for p in A3:
    x=data.loc[(data['GenHealth']==p),:].loc[data['HeartDisease']=='Yes'].count()['HeartDisease']
    y=data.loc[(data['GenHealth']==p),].count()['HeartDisease']
    B3+=[x/y]
    C3+=[y]

#Visual_7
A4=list(data.groupby('PhysicalHealth').groups.keys())
B4=[]
C4=[]
for p in A4:
    x=data.loc[(data['PhysicalHealth']==p),:].loc[data['HeartDisease']=='Yes'].count()['HeartDisease']
    y=data.loc[(data['PhysicalHealth']==p),].count()['HeartDisease']
    B4+=[x/y]
    C4+=[y]

#Visual_8
A5=list(data.groupby('SleepTime').groups.keys())
B5=[]
for p in A5:
    x=data.loc[(data['SleepTime']==p),:].loc[data['HeartDisease']=='Yes'].count()['HeartDisease']
    y=data.loc[(data['SleepTime']==p),].count()['HeartDisease']
    B5+=[x/y]
    
#Visual_9
A6=list(data.groupby('MentalHealth').groups.keys())
B6=[]
for p in A6:
    x=data.loc[(data['MentalHealth']==p),:].loc[data['HeartDisease']=='Yes'].count()['HeartDisease']
    y=data.loc[(data['MentalHealth']==p),].count()['HeartDisease']
    B6+=[x/y]

#Visual_10
B7=[]
A7=[]
for k in range(10,100,5) :
    x=data.loc[((data['BMI'] >= k) & (data['BMI']<k+5))].loc[data['HeartDisease']=='Yes'].HeartDisease.count()
    y=data.loc[(data['BMI'] >= k) & (data['BMI']<k+5)].HeartDisease.count()
    B7.append(x/y)
    A7.append(k)

#Model Machine Learning 
y=data.iloc[:,0]
X=data.iloc[:,1:]

y=y.replace('Yes',1)
y=y.replace('No',0) 

X=X.replace('Male',B0[0])
X=X.replace('Female',B0[1])

for a in range(len(A1)):
    X['AgeCategory']=X['AgeCategory'].replace(A1[a],B1[a])
for a in range(len(A2)):
    X['Race']=X['Race'].replace(A2[a],B2[a])
for a in range(len(A3)):
    X['GenHealth']=X['GenHealth'].replace(A3[a],B3[a])
for a in range(len(A4)):
    X['PhysicalHealth']=X['PhysicalHealth'].replace(A4[a],B4[a])
for a in range(len(A5)):
    X['SleepTime']=X['SleepTime'].replace(A5[a],B5[a])
for a in range(len(A6)):
    X['MentalHealth']=X['MentalHealth'].replace(A6[a],B6[a])
    
            
            
for a in range(len(A)):
    X['Diabetic']=X['Diabetic'].replace(A[a],B[a])
for l in L:
    for a in range(len(Z)):
        X[l]=X[l].replace(Z[a],D[l][a])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

y_train=y_train.replace('Yes',1)
y_train=y_train.replace('No',0)

y_test=y_test.replace('Yes',1)
y_test=y_test.replace('No',0)

accuracy_list=[]

# logistic regression
@st.cache(suppress_st_warning=True)
def logistic_regression():
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    log_reg_pred = log_reg.predict(X_test)
    log_reg_acc = metrics.accuracy_score(y_test, log_reg_pred)
    accuracy_list.append(100*log_reg_acc)
    st.write(f"logistic regression model accuracy(in %) : {log_reg_acc*100}")
    st.write(log_reg_pred)
    st.write(y_test)
    #Visualisasi Prediksi
    fig = px.scatter(
    data, x='HeartDisease', y=log_reg_pred, opacity=0.65,
    trendline='ols', trendline_color_override='darkblue'
)
    st.plotly_chart(fig)
    

# Decision Tree Classifier

def Decision_Tree():
    dt_clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')
    dt_clf.fit(X_train, y_train)
    dt_pred = dt_clf.predict(X_test)
    dt_acc = metrics.accuracy_score(y_test, dt_pred)
    accuracy_list.append(100*dt_acc)
    st.write(f'Decision Tree Classifier accuracy (in%) : {dt_acc*100}')

# K Neighbors Classifier
def KNN():
    kn_clf = KNeighborsClassifier(n_neighbors=3)
    kn_clf.fit(X_train, y_train)
    kn_pred = kn_clf.predict(X_test)
    kn_acc = metrics.accuracy_score(y_test, kn_pred)
    accuracy_list.append(100*kn_acc)
    st.write(f'K Neighbors Classifier accuracy (in%) : {kn_acc}')

# GradientBoostingClassifier
def GradientBoostClassifier():
    gradientboost_clf = GradientBoostingClassifier(max_depth=2, random_state=1)
    gradientboost_clf.fit(X_train,y_train)
    gradientboost_pred = gradientboost_clf.predict(X_test)
    gradientboost_acc = metrics.accuracy_score(y_test, gradientboost_pred)
    accuracy_list.append(100*gradientboost_acc)
    st.write(f'Gradient Boosting Classifier accuracy (in%) : {gradientboost_acc}')

# xgbrf classifier
def Xgboost():
    xgb_clf = xgboost.XGBRFClassifier(max_depth=3, random_state=1)
    xgb_clf.fit(X_train,y_train)
    xgb_pred = xgb_clf.predict(X_test)
    xgb_acc = metrics.accuracy_score(y_test, xgb_pred) 
    accuracy_list.append(100*xgb_acc)
    st.write(f'XGBoost Classifier accuracy (in%) : {xgb_acc}')

def Naivebayes():
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    n_pred = gnb.predict(X_test)
    n_acc = metrics.accuracy_score(y_test, n_pred)
    accuracy_list.append(100*n_acc)
    st.write(f'Gaussian Naive Bayes accuracy (in%) : {n_acc}')

visualization = st.sidebar.selectbox('Select visualization',('Visual_1', 'Visual_2', 'Visual_3', 'Visual_4', 'Visual_5','Visual_6', 'Visual_7', 'Visual_8', 'Visual_9', 'Visual_10'))

if visualization =='Visual_1':
    fig = px.bar(A0,B0,color=C0 , text=D0 , title="Jumlah Laki - Laki dan Perempuan yang terkena penyakit jantung")
    st.plotly_chart(fig)
    if st.button('Predict'):
        logistic_regression()

elif visualization =='Visual_2':
    fig.update_layout(height=1200, width=1200, title_text="Faktor - Faktor yang memicu penyakit jantung")
    st.plotly_chart(fig)

elif visualization == 'Visual_3':
    st.plotly_chart(px.bar(x=B,y=C,color=A , title='Terkena penyakit jantung(YES) jika diabetes(NO)', width=800 , height=1200))

elif visualization == 'Visual_4':
    st.plotly_chart(px.bar(x=C1,y=A1,color=B1 , title='Pasien yang terkena penyakit jantung berdasarkan usia', width=800 , height=800))

elif visualization == 'Visual_5':
    st.plotly_chart(px.bar(x=C2,y=A2,color=B2 , title='Jumlah Pasien terkena penyakit jantung Berdasarkan Ras', width=800 , height=800))

elif visualization == 'Visual_6':
    st.plotly_chart(px.bar(x=C3,y=A3,color=B3 , title='Jumlah Pasien terkena penyakit jantung Berdasarkan Kesehatan genetik', width=800 , height=800))

elif visualization == 'Visual_7':
    st.plotly_chart(px.line(x=A4,y=B4 , title='Probabilitas tanda - tanda terkena penyakit jantung dari hari 0 - 30 berdasarkan kesehatan fisik', width=800 , height=800))

elif visualization == 'Visual_8':
    st.plotly_chart(px.line(x=A5,y=B5 , title='Probabilitas Terkena Penyakit Jantung berdasarkan Waktu Tidur', width=800 , height=800))

elif visualization == 'Visual_9':
    st.plotly_chart(px.line(x=A6,y=B6 , title= 'Probabilitas tanda - tanda terkena penyakit jantung dari hari 0 - 30 berdasarkan kesehatan Mental', width=800 , height=800))

elif visualization == 'Visual_10':
    st.plotly_chart(px.line(x=A7,y=B7 , title='Probabilitas Terkena penyakit jantung berdasarkan BMI', width=800 , height=800))


