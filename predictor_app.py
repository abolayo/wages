import os
import pickle as pk

import pandas as pd
import streamlit as st

# Loading the model

print("Current Working Directory:", os.getcwd())  # Add this line to check the current directory
list_features = ['loc', 'title', 'bedroom', 'bathroom', 'parking_space']
with open('model_5.pkl', 'rb') as file:
    # Load the data from the pickle file
    model_classifier = pk.load(file)


# Creating a model that performs classification on user input
def app_model(input_df):
    prediction = model_classifier.predict(input_df)
    return prediction


def fancy_result(text, font_size=24):
    res = f'<span style="color:#ff0000; font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)


# Heading
def main():
    st.title('HOUSE PRICE PREDICTION APP')
    st.image('house1.jpeg')
    st.write("""
        ## Fill the form below
    """)

    # accepting the features from the user
    loc = st.number_input('Location', min_value=0, max_value=15, value=12)
    title = st.number_input('Title', min_value=1, max_value=100, value=7)
    bedroom = st.number_input('Bedroom Units', min_value=1, max_value=1000, value=10, step=5)
    bathroom = st.number_input('Bathroom Units', min_value=1, max_value=100, value=10, step=5)
    parking_space = st.number_input('Units of Parking Space', min_value=1, max_value=10, value=2)

    # st.write("""
    #     #### Model used: {}
    # """.format(model))
    # st.write()
    if st.button('CHECK'):
        input_df = pd.DataFrame(
            {'loc': [loc], 'title': [title], 'bedroom': [bedroom], 'bathroom': [bathroom],
             'parking_space': [parking_space]})

        outcome = app_model(input_df)
        st.write(f'for a house in {loc}, with {bedroom} bedrooms,'
                 f' {bathroom} bathroom and {parking_space} parking space is {outcome[0]}')


if __name__ == '__main__':
    main()
