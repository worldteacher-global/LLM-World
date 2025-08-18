import sys
sys.path.append('/home/sagemaker-user/user-default-efs/CLONED_REPOS/LLM-World/src/')
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langchain_mcp_adapters.client import MultiServerMCPClient
from mytools import MyTools
from utils import Utils
from typing import Annotated
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_aws import ChatBedrockConverse
import streamlit as st
import asyncio
import pandas as pd
import tempfile

async def StatAgent(query: str):
    load_dotenv('/home/sagemaker-user/user-default-efs/CLONED_REPOS/LLM-World/.env')

    ## Statistics Agent  ##
    client = MultiServerMCPClient(
        {
            "mcp_tools": {
                
                "url": os.getenv('MCP_URL'),
                "transport": "sse"
            }
        }
    )

    prompt = ChatPromptTemplate([
        ('system','You are a statistics agent.\n\n'
        'INSTRUCTIONS:\n'
        '- Assist ONLY with statistics-related tasks, DO NOT provide visualizations, DO NOT do any thing else\n'
        '- After you are done with your tasks, respond to the supervisor directly\n'
        '- Respond ONLY with the results of your work.'),
        ('placeholder', '{messages}')  
    ])


    mcp_tools = await client.get_tools()
    mcp_tool = None

    for obj in mcp_tools:
        if obj.name == 'google_search_tool':
            mcp_tool = obj

    mytools = MyTools()
    helper = Utils()

    cust_tools = [mytools.correlation_tool, mytools.covariance_tool, mytools.anova_one_way, mytools.levene_test, mytools.normality_tool, mytools.calculate_PC, mcp_tool]

    statistics_agent = create_react_agent(model=helper.openai_llm('gpt-4.1'), tools=cust_tools, prompt=prompt, name='statistics_agent')

    ## Visualization Agent  ##

    prompt = ChatPromptTemplate([
        ('system','You are a visualization agent.\n\n'
        'INSTRUCTIONS:\n'
        '- Assist ONLY with visualization-related tasks\n'
        '- After you are done with your tasks, respond to the supervisor directly\n'
        '- Respond ONLY with the results of your work.'),
        ('placeholder', '{messages}')  
    ])

    visualization_agent = create_react_agent(model=helper.openai_llm('gpt-4.1'), tools=[mytools.gen_plot], prompt=prompt, name='visualization_agent')

    ## Orchestration Agent  ##

    def init_handoff_tool(*, agent_name: str, description: str | None=None):
        name = f'transfer_to_{agent_name}'
        description = description or f'Ask {agent_name} for help' 

        @tool(name, description=description)
        def handoff_tool(state: Annotated[MessagesState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
            tool_message = {
                'role':'tool',
                'content': f'Successfully transferred to {agent_name}',
                'name': name,
                'tool_call_id': tool_call_id}

            
            return Command(goto=agent_name, update={**state, 'messages':state['messages']+[tool_message]}, graph=Command.PARENT)
        return handoff_tool

        
    assign_to_statistics_agent = helper.init_handoff_tool(agent_name='statistics_agent', description='Assign task to a statistics agent.')
    assign_to_visualization_agent = helper.init_handoff_tool(agent_name='visualization_agent', description='Assign task to a visualization agent.')



    prompt = ChatPromptTemplate([
        ('system','You are a supervisor managing two agents\n\n'
        '- a statistics agent. Assign statistics related tasks to this agent\n'
        '- a visualization agent. Assign visualization related tasks to this agent\n'
        'Assign work to the agents. \n'
        'Do not do any work yourself. \n'
        'Summarise the work of each agent that produced a result. \n'
        'If there is no response from one of your agents provide an explination as to why.'),    
        ('placeholder', '{messages}')  
    ])

    memory = MemorySaver()
    config = {'configurable': {'thread_id': 'test_thread'}}

    supervisor_agent = create_react_agent(model=helper.aws_llm('us.'+helper.get_aws_llms()['Claude Sonnet 4']), tools=[assign_to_visualization_agent, assign_to_statistics_agent], prompt=prompt, checkpointer=memory, name='supervisor')

    # supervisor_agent = create_react_agent(model=helper.openai_llm('gpt-4.1'), tools=[assign_to_visualization_agent, assign_to_statistics_agent], prompt=prompt, checkpointer=memory, name='supervisor')

    ### Orchestration Flow  ###

    supervisor = (
        StateGraph(MessagesState)
        .add_node(supervisor_agent, destinations=('statistics_agent', 'visualization_agent', END))
        .add_node(statistics_agent)
        .add_node(visualization_agent)
        .add_edge(START, 'supervisor')
        .add_edge('statistics_agent', 'supervisor')
        .add_edge('visualization_agent', 'supervisor')
        .compile()
    )

    ## Run to completion
    final_response = await supervisor.ainvoke({'messages':[{'role':'user','content': [{'type': 'text', 'text':query}]}]}, config)

    return final_response['messages'][-1].content


if __name__=='__main__':

    async def main():
        st.title('I am a Statistics Assistant')   

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Submit a question."):

            st.chat_message("user").markdown(prompt)
        
            st.session_state.messages.append({"role": "user","content":prompt})
            
            with st.chat_message("assistant"):
                
                response = await StatAgent(prompt)
                # st.markdown(response_generator(response))
                
                st.markdown(response)
            
                st.session_state.messages.append({"role":"assistant", "content":response})       

        
        ## logic for file uploads     

        file_uploaded = st.file_uploader("Upload a file.", type="csv")

        if file_uploaded:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir,file_uploaded.name)

                with open(temp_file_path, "wb") as f:
                    f.write(file_uploaded.getbuffer())
                
                st.success(f"File was saved at: {temp_file_path}")
                st.write(temp_fle_path.name)

    asyncio.run(main())