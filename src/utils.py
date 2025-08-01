from langchain_aws import ChatBedrockConverse # Langchain LLM object
from langchain_openai import AzureChatOpenAI # Langchain LLM object
from langchain_core.messages import convert_to_messages
import boto3
import requests
import os

class Utils:
    ################################################# CUSTOM FUNCTIONS ############################################################
    
    @staticmethod
    def clean_print_msgs(update, last_message=False):
        '''Clean print llm output'''
        is_subgraph = False
        if isinstance(update, tuple):
            ns, update = update
            # skip parent graph updates in the printouts
            if len(ns) == 0:
                return
            
            graph_id = ns[-1].split(':')[0]
            print(f'Update from subgraph {graph_id}:')
            print('\n')
            is_subgraph = True

        for node_name, node_update in update.items():
            update_label = f'Update from node {node_name}:'
            if is_subgraph:
                update_label = '\t' + update_label
            print(update_label)
            print('\n')

            messages = convert_to_messages(node_update['messages'])
            if last_message:
                messages = messages[-1:]

            for m in messages:
                # clean_print_msg(m, indent=is_subgraph)

                clean_message = m.pretty_repr(html=True)
                indent=is_subgraph
                if not indent:
                    print(clean_message)
                    return
                indented = '\n'.join('\t' + c for c in clean_message.split('\n'))
                print(indented)
            print('\n')


    @staticmethod
    def aws_llm(modelid: str):  
        '''Create LangChain Object for AWS Bedrock llm'''     

        return ChatBedrockConverse(model_id=modelid, region_name='us-east-1')

    @staticmethod
    def openai_llm(modle_id: str):
        '''Create LangChain LLM object for Azure OpenAI llm'''
        api_response = requests.get(os.getenv('AZURE_OPENAI_BASEURL'))
        payload = api_response.json()

        gllm = AzureChatOpenAI(
            deployment_name=modle_id,  
            openai_api_version="2024-12-01-preview",
            azure_endpoint=payload['nonprod'][modle_id][0]['endpoint'],
            api_key=os.getenv('AZURE_OPENAI_KEY'),
            temperature=0)
        return gllm

    @staticmethod
    def get_openai_llms():
        '''get a list of llms available in Azure OpenAI'''
        response = requests.get(os.getenv('AZURE_OPENAI_BASEURL'))
        returned_payload = response.json()

        return [models for models in returned_payload['nonprod'].keys()]

    @staticmethod
    def get_aws_llms():
        '''get a list of llms available in AWS Bedrock'''
        bedrock_client = boto3.client('bedrock', region_name='us-east-1')

        response = bedrock_client.list_foundation_models()

        llm_dict = {}
        for obj in  response['modelSummaries']:
            llm_dict[obj['modelName']] = obj['modelId']

        return llm_dict