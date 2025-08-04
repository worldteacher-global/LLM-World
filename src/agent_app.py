# Multi-Agent Framework
import streamlit as st
from dotenv import load_dotenv
load_dotenv('/home/ec2-user/BoW/LLM-World/.env')
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain_aws import ChatBedrockConverse
import sys
sys.path.append('/home/ec2-user/BoW/LLM-World/src/')
from mytools import MyTools
from utils import Utils
import asyncio
import os
## Handoff tool packages
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command

##
async def _multiAgent(user_input: str):
    ## Statistics Agent
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
    ## Visualization Agent
    prompt = ChatPromptTemplate([
        ('system','You are a visualization agent.\n\n'
        'INSTRUCTIONS:\n'
        '- Assist ONLY with visualization-related tasks\n'
        '- After you are done with your tasks, respond to the supervisor directly\n'
        '- Respond ONLY with the results of your work.'),
        ('placeholder', '{messages}')  
    ])
    visualization_agent = create_react_agent(model=helper.openai_llm('gpt-4.1'), tools=[mytools.gen_plot], prompt=prompt, name='visualization_agent')
   ## Handoff tool/ tool creation  
    assign_to_statistics_agent = helper.init_handoff_tool(agent_name='statistics_agent', description='Assign task to a statistics agent.')
    assign_to_visualization_agent = helper.init_handoff_tool(agent_name='visualization_agent', description='Assign task to a visualization agent.')
    ## Supervisor Agent
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
    config = {'configurable': {'thread_id': 'streamlit_thread'}} # Using a consistent thread_id for the session
    supervisor_agent = create_react_agent(model=helper.aws_llm('us.'+helper.get_aws_llms()['Claude Sonnet 4']), tools=[assign_to_visualization_agent, assign_to_statistics_agent], prompt=prompt, checkpointer=memory, name='supervisor')
  #  supervisor_agent = create_react_agent(model=helper.openai_llm('gpt-4.1'), tools=[assign_to_visualization_agent, assign_to_statistics_agent], prompt=prompt, checkpointer=memory, name='supervisor')
    
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

    # MODIFIED SECTION: Instead of just printing, we capture the final response to return it.
    final_response = "Agent processing complete, but no final summary was generated."  # Default message
    
    async for chunk in supervisor.astream(
        {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': user_input}]}]},
        config=config
    ):
        # The supervisor is designed to summarize the work. We look for its last message
        # that is not a tool call, as this will be the final answer.
        if "supervisor" in chunk:
            last_message = chunk["supervisor"]["messages"][-1]
            if last_message.content and not last_message.tool_calls:
                final_response = last_message.content

    return final_response

def multiAgent(user_query: str):
    ''' Run logic in a blocking way'''
    return asyncio.run(_multiAgent(user_query))

if __name__=='__main__':
    st.title("Multi-Agent Analysis Framework")
    user_input = st.text_input("Hello! Please submit a question or request:")
    
    if user_input:
        # Add a spinner to give the user feedback while the agent is running
        with st.spinner("Agents are collaborating on your request..."):
            result = multiAgent(user_input)
        
        st.subheader("Agent Response")
        
        # Use st.markdown to properly render any formatting (like lists or bold text)
        if result:
            st.markdown(result)
        else:
            st.warning("The agent process completed, but no output was returned.")

