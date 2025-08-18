# ChatBot Generative AI
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests
import time
import random

load_dotenv('/home/ec2-user/BoW/LLM-World/.env')

def chat_gpt_41(ask_a_question):
    
    api_response = requests.get(os.getenv('AZURE_OPENAI_BASEURL'))
    payload = api_response.json()

    endpoint = payload['nonprod']['gpt-4.1'][0]['endpoint']

    client = AzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    api_version = '2024-12-01-preview',
    azure_endpoint = endpoint)

    convo = []
    convo.append({'role':'user','content':ask_a_question})
    response = client.chat.completions.create(
    model = 'gpt-4.1',
    max_tokens = 30000,
    messages = convo,
    temperature = 0)

    reply = response.choices[0].message.content
    convo.append({'role':'assistant','content':reply})

    return reply

# From streamlit doc    
def response_generator(response):
    # response = random.choice(
    #     [
    #         "Hello there! How can I assist you today?",
    #         "Hi, human! Is there anything I can help you with?",
    #         "Do you need help?",
    #     ]
    # )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

if __name__=='__main__':
    
    st.title('Just a General ChatBot')

   

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Submit a question."):

        st.chat_message("human").markdown(prompt)
        # with st.chat_message("human"):
        #     st.markdown(prompt)        
        st.session_state.messages.append({"role": "human","content":prompt})
        
        with st.chat_message("ai"):
            
            response = " ".join(chat_gpt_41(prompt))
            
            # st.markdown(response_generator(response))
            st.markdown(chat_gpt_41(prompt))
        
        st.session_state.messages.append({"role":"ai", "content":chat_gpt_41(prompt)})       

    
    ## logic for file uploads
    import pandas as pd

    file_uploaded = st.file_uploader("Upload a file.", type="csv")

    if file_uploaded:
        st.write(file_uploaded.name)
        # dataframe = pd.read_csv(file_uploaded)
        # st.dataframe(dataframe)


            

 





    
