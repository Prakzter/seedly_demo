import streamlit as st
import pandas as pd
import numpy as np
import pickle
import textcleaner
import sklearn




### code for frontend ###

st.image("data:image/png;base64,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", use_column_width=True)
st.title("Stocks or Real Estate investing?")
st.subheader("This app aims to serve as a simple frontend to demonstrate our text classification model.")


st.write("---"*20)
'''#### Don't have any text to try out? Copy our sample text here and analyse below...
   ####  
'''


if st.checkbox('Sample text for Stocks Investment'):
    st.write(""" I am currently using StashAway as my main investment. I wish to try stocks/ETF index funds, any suggestions on where I should invest?""")

if st.checkbox('Sample text for Real Estate Investment'):
    st.write(""" At a young age (e.g 23 years old) with Low income (<$4000), is it advisable to venture 
    commercial and industrial property investment instead, and use the profits to fund residential property?""")

if st.checkbox('Sample text for mixed class'):
    st.write('''There's a huge undeniable bubble going on in the stock market, it's just a question how about long further it will stretch till it burst.
But I'm in a dilemma, I found a nice coming apartment that will be completed in 6-7 months. It's slightly overpriced, but not too bad. How ever, is it a terrible idea to buy an apartment right now?
Not sure what market crash will result. Stocks will decline alot, but how about real estate?
It's also said that pricing on housing/apartment is all time high in my country.''')
st.write("---"*20)

st.subheader('Enter your question here:')
message = st.text_area('',"""input here...""")

# code to transform msg
topics = ['Real Estate Investment', 'Stocks Investment']

cleaned_text = textcleaner.transform_text(message)
if st.checkbox('Show cleaned text (removing non-word characters,digits and lemmatise text)'):
    st.write(cleaned_text[0])

def predict(text_list):
    model = pickle.load(open('seedly.pkl','rb'))
    return model.predict(text_list)


def predict_proba(text_list):
    model = pickle.load(open('seedly.pkl','rb'))
    return model.predict_proba(text_list)


if st.button('Analyse post'):
    confidence_0 = round(predict_proba(cleaned_text)[0][0],3)
    confidence_1 = round(predict_proba(cleaned_text)[0][1],3)
    st.write("The probability that it should be classified as Real Estate" , confidence_0)
    st.write("The probability that it should be classified as Stocks" , confidence_1)
    
    if confidence_0 > 0.80 or confidence_1 > 0.80:
        st.balloons()
    
    prediction = predict(cleaned_text)[0]
    st.text('---'*20)
    st.write('\n\nGreat this post should belong to:')
    st.text(topics[prediction])

    

    


