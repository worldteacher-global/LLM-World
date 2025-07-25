# ChatBot Generative AI
import streamlit as st

from openai import AzureOpenAI

import os
import requests

endpoints = '<Enter-Azure-OpenAI-BaseURL>'

api_response = requests.get(endpoints)
payload = api_response.json()
payload['nonprod'].keys()

endpoint = payload['nonprod']['gpt-4.1'][0]['endpoint']

os.environ['AZURE_OPENAI_KEY'] = '<Enter-Azure-OpenAI-API-Key>'

client = AzureOpenAI(
api_key=os.getenv('AZURE_OPENAI_KEY'),
api_version = '2024-12-01-preview',
azure_endpoint = endpoint)

def chat_gpt_41(ask_a_question):
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

    user_input = st.text_input("Submit a question or comment!!")

    st.title(chat_gpt_41(user_input))

    
