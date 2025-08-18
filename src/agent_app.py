# Multi-Agent Framework
import streamlit as st
from dotenv import load_dotenv
load_dotenv('/home/ec2-user/BoW/LLM-World/.env')
from langchain_core.prompts import ChatPromptTemplate
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
import re 
from typing import Tuple
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState, END
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

import tempfile

# Final Answer Tool (unchanged)
@tool
def final_answer(answer: str) -> str:
    """Use this tool to provide the final, consolidated answer to the user."""
    return answer

# Robust Supervisor Router (unchanged)
def supervisor_router(state: MessagesState):
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        if last_message.tool_calls[0]['name'] == 'final_answer':
            return END
        return [tool_call['name'] for tool_call in last_message.tool_calls]
    else:
        return END

# State Normalizer (unchanged)
def normalize_messages(state: MessagesState) -> MessagesState:
    new_messages = []
    for msg in state['messages']:
        if isinstance(msg, dict):
            if msg.get("role") == "ai" and msg.get("tool_calls"): new_messages.append(AIMessage(**msg))
            elif msg.get("role") == "tool": new_messages.append(ToolMessage(**msg))
            else: new_messages.append(AIMessage(content=str(msg.get("content", ""))))
        else: new_messages.append(msg)
    state['messages'] = new_messages
    return state

async def _multiAgent(user_input: str) -> Tuple[str, str | None]:
    client = MultiServerMCPClient({"mcp_tools": {"url": os.getenv('MCP_URL'), "transport": "sse"}})
    mcp_tools = await client.get_tools()
    mcp_tool = next((obj for obj in mcp_tools if obj.name == 'google_search_tool'), None)
    
    mytools = MyTools()
    helper = Utils()
    
    prompt_stats = ChatPromptTemplate.from_messages([('system', 'You are a statistics agent...'), ('placeholder', '{messages}')])
    cust_tools = [mytools.correlation_tool, mytools.covariance_tool, mytools.anova_one_way, mytools.levene_test, mytools.normality_tool, mytools.calculate_PC, mcp_tool]
    statistics_agent = create_react_agent(model=helper.openai_llm('gpt-4.1'), tools=cust_tools, prompt=prompt_stats, name='statistics_agent')
    
    prompt_viz = ChatPromptTemplate.from_messages([('system', 'You are a visualization agent... Your final response MUST be ONLY the full, prefixed base64 string...'), ('placeholder', '{messages}')])
    visualization_agent = create_react_agent(model=helper.openai_llm('gpt-4.1'), tools=[mytools.gen_plot], prompt=prompt_viz, name='visualization_agent')
    
    # --- MODIFIED: Create highly descriptive handoff tools ---
    assign_to_statistics_agent = helper.init_handoff_tool(
        agent_name='statistics_agent', 
        description='Use this to delegate tasks involving statistical calculations, analysis, or explanations of concepts like normality, correlation, or p-values to the statistics_agent.'
    )
    assign_to_visualization_agent = helper.init_handoff_tool(
        agent_name='visualization_agent', 
        description='Use this to delegate tasks involving creating plots, charts, and graphs like scatter plots, line plots, and bar charts to the visualization_agent.'
    )
    
    # --- MODIFIED: Enhance the supervisor's prompt with explicit instructions ---
    prompt_sup = ChatPromptTemplate.from_messages([
        ('system', 'You are a supervisor managing two agents. Your primary role is to understand the user\'s request and delegate it to the correct agent.\n\n'
                   'Here are your agents and their capabilities:\n'
                   '- **visualization_agent**: Use this for any requests related to creating plots, charts, or graphs (e.g., "create a scatter plot").\n'
                   '- **statistics_agent**: Use this for requests involving statistical calculations or explanations (e.g., "what is normality?", "calculate the correlation").\n\n'
                   'After an agent completes its work, you will receive the results. Collate all results.\n'
                   'When all work is done, or if the user asks a simple question you can answer directly (like "hello"), you MUST use the `final_answer` tool to provide the complete response.'),
        ('placeholder', '{messages}')
    ])
    
    memory = MemorySaver()
    config = {'configurable': {'thread_id': 'streamlit_thread_v10_descriptive'}} 
    
    supervisor_tools = [assign_to_visualization_agent, assign_to_statistics_agent, final_answer]
    supervisor_agent = create_react_agent(model=helper.aws_llm('us.'+helper.get_aws_llms()['Claude Sonnet 4']), tools=supervisor_tools, prompt=prompt_sup, checkpointer=memory, name='supervisor')
    
    graph = StateGraph(MessagesState)
    graph.add_node("supervisor", supervisor_agent); graph.add_node("statistics_agent", statistics_agent); graph.add_node("visualization_agent", visualization_agent); graph.add_node("normalize_state", normalize_messages)
    graph.add_conditional_edges("supervisor", supervisor_router, {"assign_to_statistics_agent": "statistics_agent", "assign_to_visualization_agent": "visualization_agent", END: END})
    graph.add_edge(START, "supervisor"); graph.add_edge("statistics_agent", "normalize_state"); graph.add_edge("visualization_agent", "normalize_state"); graph.add_edge("normalize_state", "supervisor")
    supervisor = graph.compile(checkpointer=memory)

    final_text_response = "Agent processing complete, but no final summary was generated."
    base64_image_data = None
    
    async for chunk in supervisor.astream({'messages': [{'role': 'user', 'content': user_input}]}, config=config):
        if "supervisor" in chunk:
            messages = chunk["supervisor"].get("messages", [])
            for message in messages:
                # The final answer is now the content of the `final_answer` ToolMessage.
                if isinstance(message, ToolMessage) and message.name == "final_answer":
                    content = message.content
                    # Check if the final answer also contains a base64 string
                    if "base64_image:" in content:
                        parts = content.split("base64_image:", 1)
                        # Provide a nice default message if there's no other text
                        final_text_response = parts[0].strip() if parts[0].strip() else "A visualization has been generated for you."
                        base64_image_data = parts[1]
                    else:
                        final_text_response = content

    return final_text_response, base64_image_data

def multiAgent(user_query: str) -> Tuple[str, str | None]:
    return asyncio.run(_multiAgent(user_input=user_query))



if __name__=='__main__':
    st.set_page_config(layout="wide")
    st.title("Multi-Agent Analysis Framework")
    if 'messages' not in st.session_state: st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("image"): st.image(message["image"], caption="Generated Visualization")
    if user_input := st.chat_input("Hello! Please submit a question or request:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.spinner("Agents are collaborating on your request..."):
            text_result, image_to_display = multiAgent(user_input)
        with st.chat_message("assistant"):
            if text_result:
                st.markdown(text_result)
                msg_to_store = {"role": "assistant", "content": text_result}
            else:
                st.warning("The agent process completed, but no text output was returned.")
                msg_to_store = {"role": "assistant", "content": "No text output was generated."}
            if image_to_display:
                st.image(image_to_display, caption="Generated Visualization")
                msg_to_store["image"] = image_to_display
            st.session_state.messages.append(msg_to_store)


      ## logic for file uploads     

    file_uploaded = st.file_uploader("Upload a file.", type="csv")

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if file_uploaded:
        for file in file_uploaded:
            file_path = os.path.join(UPLOAD_FOLDER,file.name)
            with open(temp_file_path, "wb") as f:
                f.write(file.getbuffer())
            
            st.success(f"File was saved at: {file_path}")

