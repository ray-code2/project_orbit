#import Libraries yang diperlukan
from matplotlib.pyplot import text
import streamlit as st
import pandas as pd
import numpy as np
from turtle import width
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Membangun aplikasi dashboard
# image = Image.open("Health.jpeg")
#st.image(image, width=500)

# Import Dataset
@st.cache
def load_data():
    df = pd.read_csv("cd.csv")
    return df

df = load_data()

#VARIABEL

visualization = st.sidebar.selectbox('Select visualization',('Home_Navigation','Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5','Prediction'))


if visualization =='Home_Navigation':
    my_bar = st.progress(0)
    'Please wait Patiently...'
    for percent_complete in range(100):
      
      time.sleep(0.01)
      timer = my_bar.progress(percent_complete + 1)
      
  
    st.header("Exploratory Data Analysis Causes of Death☠️ - Our World in Data Dataset")
    
elif visualization == 'Model_1':
    column_names = ["Entity","Code","Year","Causes name","Causes Full Description","Death Numbers"]
    selected_state = df.reindex(columns=column_names)
    state_select = st.sidebar.selectbox('Select a Country', df['Entity'].unique())
    selected_state = df[df['Entity'] == state_select]
    fig1 = px.bar(selected_state, x='Year', y='Death Numbers',
    hover_data=['Code', 'Causes Full Description'], color='Causes name',
    labels={'Death Numbers':'Total Death'}, height=600 , width=1300 ,
    )
    
    st.title(f'Total Death by diseases & accidents in {state_select} From 1990 - 2019')
    st.plotly_chart(fig1)
elif visualization == 'Model_2':
    column_names = ["Entity","Code","Year","Causes name","Causes Full Description","Death Numbers"]
    selected_state = df.reindex(columns=column_names)
    state_select = st.sidebar.selectbox('Select a Country', df['Entity'].unique())
    selected_state = df[df['Entity'] == state_select]
    Cause_select = st.sidebar.selectbox('Select Death Cause' , selected_state['Causes name'].sort_values(ascending=True).unique())
    selected_cause = selected_state[selected_state['Causes name'] == Cause_select]
    st.title(f'Country : {state_select}')
    df1 = selected_cause.sort_values(by=['Year'])
    df2 = df1.replace(np.nan,0)
    df3=df2.pivot_table(index=['Causes name'],values=['Death Numbers'],aggfunc=sum).iloc[[0],[0]].values
    fig2 = px.bar(df2, x='Causes name', y='Death Numbers',
    hover_data=['Code', 'Death Numbers','Year'],color='Causes name',
    labels={'Death Numbers':'Total Death'}, height=400 , width=400)
    st.header(f"Total deaths Caused by {Cause_select} From 1990 - 2019 : {str(int(df3))}")
    st.plotly_chart(fig2)
elif visualization == 'Model_3':
    column_names = ["Entity","Code","Year","Causes name","Causes Full Description","Death Numbers"]
    selected_state = df.reindex(columns=column_names)
    state_select = st.sidebar.selectbox('Select a Country', df['Entity'].unique())
    selected_state = df[df['Entity'] == state_select]
    year_select = st.sidebar.selectbox('Select Year' , df['Year'].sort_values(ascending=True).unique())
    selected_year = selected_state[selected_state['Year'] == year_select]
    df4=selected_year.pivot_table(index=['Year'],values=['Death Numbers'],aggfunc=sum).iloc[[0],[0]].values
    fig3 = px.bar(selected_year, x='Year', y='Death Numbers',
    hover_data=['Code', 'Death Numbers'],color='Causes name',
    labels={'Death Numbers':'Total Death by diseases & accidents'}, height=400, width=650,
    )
    st.title(f'Total Deaths by diseases & accidents in {state_select} in {year_select} : {str(int(df4))}')
    st.plotly_chart(fig3)
   
elif visualization == 'Model_4':
    column_names = ["Entity","Code","Year","Causes name","Causes Full Description","Death Numbers"]
    selected_state = df.reindex(columns=column_names)
    state_select = st.sidebar.selectbox('Select a Country', df['Entity'].unique())
    selected_state = df[df['Entity'] == state_select]
    Cause_select = st.sidebar.selectbox('Select Death Cause' , selected_state['Causes name'].sort_values(ascending=True).unique())
    selected_cause = selected_state[selected_state['Causes name'] == Cause_select]
    df1 = selected_cause.sort_values(by=['Year'])
    df2 = df1.replace(np.nan,0)
    df3=df2.pivot_table(index=['Causes name'],values=['Death Numbers'],aggfunc=sum).iloc[[0],[0]].values
    fig4 = px.line(df2,x='Year' , y='Death Numbers'
    ,hover_data=['Code', 'Causes name'],
    )
    st.title(f'Total deaths Caused by {Cause_select} in {state_select} From 1990 - 2019 : {str(int(df3))}')
    st.plotly_chart(fig4)

elif visualization == 'Model_5':
    column_names = ["Entity","Code","Year","Causes name","Causes Full Description","Death Numbers"]
    selected_state = df.reindex(columns=column_names)
    state_select = st.sidebar.selectbox('Select a Country', df['Entity'].unique())
    selected_state = df[df['Entity'] == state_select]
    year_select = st.sidebar.selectbox('Select Year' , df['Year'].sort_values(ascending=True).unique())
    selected_year = selected_state[selected_state['Year'] == year_select]
    fig5 = px.bar(selected_year, x='Causes name', y='Death Numbers', text='Causes name',
    hover_data=['Code', 'Death Numbers'],color='Causes name',
    labels={'Death Numbers':'Total Death by diseases & accidents'}, height=800, width=1250,
    )
    st.title(f'Deaths in {state_select} From {year_select}')
    st.plotly_chart(fig5)


elif visualization == 'Prediction':
    'Select a Country you want to predict :'
    column_names = ["Entity","Code","Year","Causes name","Causes Full Description","Death Numbers"]
    selected_state = df.reindex(columns=column_names)
    state_select = st.sidebar.selectbox('Select a Country', df['Entity'].unique())
    selected_state = df[df['Entity'] == state_select]
    Cause_select = st.sidebar.selectbox('Select Death Cause' , selected_state['Causes name'].sort_values(ascending=True).unique())
    selected_cause = selected_state[selected_state['Causes name'] == Cause_select]
    df1 = selected_cause.sort_values(by=['Year'])
    df2 = df1.replace(np.nan,0)
    df3=df2.pivot_table(index=['Causes name'],values=['Death Numbers'],aggfunc=sum).iloc[[0],[0]].values
    # Memberikan Label Machine learning untuk menentukan Jika angka kematian 
    # disebabkan oleh penyakit tertentu == 0 maka beri label "0" jika ada beri label "1"
    label = []
    for index, row in df2.iterrows():
        if row["Death Numbers"] == 0:
            label.append(0)
        else:
            label.append(1)

    df2["label"] = label

    st.write(df3)