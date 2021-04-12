
from tensorflow.keras.models import load_model
import funcs
import streamlit as st
from PIL import Image

translations = {'Al götür': 'Take Away'}

# Creating the Titles and Image

st.title("English Spanish Translator")
st.header("Bringing You The Most Innacurate Translations :sunglasses:")
st.write("Seriously, close this and open Google Translate")
image = Image.open('cocomex1.jpg')
image2 = Image.open('cocomex2.jpg')

col1, col2 = st.beta_columns(2)
with col1:
    st.image(image, use_column_width=True)
with col2:
    st.image(image2, use_column_width=True)

with st.sidebar:
    option1 = st.selectbox('Select Original Language',('Spanish', 'English'))
    option2 = st.selectbox('Select Destination Language',('Spanish', 'English'))    

    user_input = st.text_input("Choose your Phrase!", 'type it here :)')

    if st.button('Translate'):
          if option1 == 'Spanish' and option2 == 'English':
               output = funcs.getEngTrans(user_input)
               st.write(" English Translation:\n", output)
          elif option2 == 'Spanish' and option1 == 'English':
               output = funcs.getSpaTrans(user_input)
               st.write(" Spanish Translation:\n", output)
          else:
               st.write(" original and destination languages can't be the same!")
     
     #if st.button('View Actual Translation'):
     #     try:
     #          output2 = translations[user_input]
     #          st.write(" Actual English Translation:\n", output2)
     #     except KeyError: 
     #         st.write("I don't have an actual translation for that :(")
     
