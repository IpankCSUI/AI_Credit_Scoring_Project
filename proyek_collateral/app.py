import streamlit as st
import pandas as pd

# Title of the app
st.title("Hello Streamlit")

name = st.text_input("Enter your name")
st.write(f"Hello, {name}!")

age = st.slider("Select your age", 0, 100, 25)
st.write(f"Your age is {age}")

agree = st.checkbox("I agree")
if agree:
    st.write("Great!")

option = st.selectbox(
    "Which number do you like best?",
    [1, 2, 3, 4, 5]
)
st.write(f"You selected: {option}")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
st.pyplot(fig)

import seaborn as sns

df = sns.load_dataset("iris")
fig = sns.pairplot(df)
st.pyplot(fig)

import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
st.plotly_chart(fig)



