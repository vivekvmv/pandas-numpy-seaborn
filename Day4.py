import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('iris.csv')
a=np.max(df)

st.title("ML ICFOSS")
st.header("Assignment 4-Pandas,Numpy & Seaborn")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader("Iris Dataset")
st.write(df)

st.subheader("Maximum values in each attributes")
st.write(a)

st.subheader("Scatter plotting of species with attributes sepel_length,sepel_width,petel_length and petel_width")
st.write(sns.pairplot(df,hue='species',palette="muted",size=5,vars=['sepal_width','sepal_length','petal_length','petal_width'],kind='scatter'))
st.pyplot()

st.subheader("Violin plot with attributes species and sepel_width")
st.write(sns.violinplot(x='species',y='sepal_width',data=df))
st.pyplot()

st.subheader("Boxplot of the attribute sepal_length")
st.write(sns.boxplot(y='sepal_length', data=df))
st.pyplot()
