import streamlit as st
import requests

st.write("Identify the category of goods by a review")

# Create text input fields
title = st.text_input("Title")
text = st.text_input("Text")

# Create a submit button
if st.button("Submit"):
    # Make a GET request with the text as parameters
    response = requests.get(f"http://fastapi:8000/predict/title={title}&text={text}")

    # Display the response text
    st.write("Predicted category: "+response.json()['result'])
