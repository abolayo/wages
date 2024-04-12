import streamlit as st
import pickle as pk
import os

# Loading the model

print("Current Working Directory:", os.getcwd())  # Add this line to check the current directory
with open('model_5.pkl', 'rb') as file:
    # Load the data from the pickle file
    try:
        model_classifier = pk.load(file)
    except Exception as e:
        print("Error loading the file:", str(e))

list_features = ['loc', 'title', 'bedroom', 'bathroom', 'parking_space']


# Creating a model that performs knn classification on user input
def model(loc, title, bedroom, bathroom, parking_space):
    features = [[loc, title, bedroom, bathroom, parking_space]]

    prediction = model_classifier.predict(features)
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
    outcome = model(loc, title, bedroom, bathroom, parking_space)
    st.write(f'for a house in {loc}, with {bedroom} bedrooms, {bathroom} bathroom and {parking_space} parking space is {outcome}')
    # Function to display output
    # def result(outcome):
    #     if outcome == 1:
    #         return fancy_result('You have been diagnosed with Diabetes🤒')
    #         st.write('Check out our recommendations page to book an appointment with a doctor near you')
    #     else:
    #         return st.write("""
    #     ### You have not been diagnosed with diabetes
    # """)
    #
    # # Function for recommendation
    # def recommend(outcome):
    #     if outcome == 1:
    #
    #         st.write('Check out our recommendations page to book an appointment with a doctor near you')
    #     else:
    #         return st.write('Check out our recommendations page for Health and Diet tips')
    #
    # st.success(outcome)
    # result(outcome)
    # recommend(outcome)


if __name__ == '__main__':
    main()
