import os
import sys
sys.path.append('/home/sagemaker-user/user-default-efs/CLONED_REPOS/LLM-World/src')
from mytools import MyTools
from utils import Utils
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv
load_dotenv('/home/sagemaker-user/user-default-efs/CLONED_REPOS/LLM-World/.env')
import streamlit as st

def oneshotagent(input_prompt: str):
    prompt = input_prompt
    
    mcp_client = (
        {
            "mcp_tools":{
                "url":os.getenv("MCP_URL"),
                "transport":"sse"
            }
        }
    )

    system_prompt = ChatPromptTemplate(
        [
            (
                "system","You are a statistics agent. \n\n"
                "INSTRUCTIONS:\n"
                "- Assist with statistics related tasks. \n\n"
                "- Call the apprpriate tools to complete a task. \n\n"
                " List your reasoning when completing a task. \n\n"
                "- Respond with the results of your work, explaning the final result in clear detail."
            ),
            (
                "placeholder", "{messages}"
            )
        ]
    )

    tools = MyTools()
    helper = Utils()

    # mcp_tools = await mcp_client.get_tools()
    # mcp_tool = None
    # for tool_obj in mcp_tools:
    #     if tool_obj.name == "google_search_tool":
    #         mcp_tool = tool_obj 

    tools = [tools.correlation_tool, tools.covariance_tool, tools.anova_one_way, tools.levene_test, tools.normality_tool, tools.calculate_PC, tools.gen_plot, tools.display_base64_image]

    agent = create_react_agent(
        model=helper.openai_llm("gpt-4.1"),
        tools=tools,
        prompt=system_prompt,
        name="statistics_agent" 
    )

    agent_response = agent.invoke(
        {"messages": [
            {"role":"user", "content":[{"type":"text", "text":prompt}]}
            ]
        }
    )
    
    return agent_response

if __name__=='__main__':
    # st.
    query = 'Generate a line plot with the title "Sales Over Time. The x-axis should be [1, 2, 3, 4, 5] representing months, and the y-axis should be [10, 20, 15, 30, 25] representing sales in thousands.'
    
    result = oneshotagent(query)

    from langchain_core.messages import ToolMessage 

    image_path = None

    for obj in result['messages']:
        if isinstance(obj, ToolMessage):     
            if obj.name=='gen_plot':
                image_path=obj.content

    print(image_path)