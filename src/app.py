# ChatBot Generative AI
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests

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


if __name__=='__main__':

    user_input = st.chat_input("Submit a question or comment!!")

    # st.title(chat_gpt_41(user_input))
    st.write(chat_gpt_41(user_input))

    
