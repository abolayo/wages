import os
import pickle as pk
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Loading the model

print("Current Working Directory:", os.getcwd())  # Add this line to check the current directory
list_features = ['loc', 'title', 'bedroom', 'bathroom', 'parking_space']
with open('model_5.pkl', 'rb') as file:
    # Load the data from the pickle file
    model_classifier = pk.load(file)


# Creating a model that performs classification on user input
def app_model(input_df):
    prediction = np.exp(model_classifier.predict(input_df))
    return prediction


def wrangle(data):
    # Read the file into a dataframe


    # Fill in the missing values for both variable types
    for name in data.select_dtypes("number"):
        data[name] = data[name].fillna(value=np.mean(data[name]))
    for name in data.select_dtypes("object"):
        data[name] = data[name].fillna("None")

    # collect categorical features into a list
    cat_columns = data.dtypes[data.dtypes == "object"].index.to_list()

    # Use label encoder to encode the categorical features
    for c in cat_columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(list(data[c].values))
        data[c] = label_encoder.transform(list(data[c].values))
    return data


# Heading
def main():
    st.title('HOUSE PRICE PREDICTION APP')
    st.image('house1.jpeg')
    st.write("""
        ## Fill the form below
    """)

    # accepting the features from the user
    loc = (
        "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno", "Cross River",
        "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi",
        "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun",
        "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara",
        "Territory Federal Capital Territory",
    )
    title = ('Semi-detached duplex', 'Apartment', 'Detached duplex', 'Terrace duplex',
             'Mansion', 'Bungalow', 'Penthouse', 'Townhouse', 'Flat', 'Cottage',
             )
    loc = st.selectbox("State", loc)
    title = st.selectbox("Type", title)
    bedroom = st.number_input('Bedroom Units', min_value=1, max_value=10, value=1, step=1)
    bathroom = st.number_input('Bathroom Units', min_value=1, max_value=7, value=1)
    parking_space = st.number_input('Units of Parking Space', min_value=1, max_value=6, value=1)

    # st.write("""
    #     #### Model used: {}
    # """.format(model))
    # st.write()
    if st.button('CHECK'):
        input_df = pd.DataFrame(
            {'loc': [loc], 'title': [title], 'bedroom': [bedroom], 'bathroom': [bathroom],
             'parking_space': [parking_space]})
        #input_df.to_csv('input_df.csv')
        dataframe = wrangle(input_df)
        outcome = np.exp(app_model(dataframe))
        st.write(f'For a  beautifully finished house in the city of {loc}, with {bedroom} bedrooms,'
                 f' {bathroom} bathroom and {parking_space} parking space the estimated cost is N{outcome[0]:.2f}')


if __name__ == '__main__':
    main()
